#include "preprocessor.h"

#include <cmath>
#include <thread>

#include "face.h"
#include "image4d.h"

using cv::Vec3f;
using std::string;
using std::vector;

namespace face {

enum class ScanOrder {
    top_down,
    bottom_up,
    left_to_right,
    right_to_left
};

/**
 * @brief gives the index of the first row or column with at least minNonemptySquares
 *        squares different != 0
 * The matrix is scanned according to the specified scanOrder wich establishes if
 * it is scanned by row or by column and if by increasing or decreasing indexes
 *
 * @note template T specifies the data type stored in the Mat
 */
template<typename T>
int getFirstNonempty(cv::Mat matrix, int minNonemptySquares, ScanOrder scanOrder);


const string Preprocessor::FACE_DETECTOR_PATH = "../haarcascade_frontalface_default.xml";
const string Preprocessor::POSE_ESTIMATOR_PATH = "../trees/";

// ---------- constructors ----------

Preprocessor::Preprocessor(const string& faceDetectorPath, const string& poseEstimatorPath)
{
    // load the pretrained face detection model
    classifier = cv::CascadeClassifier(faceDetectorPath);
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return;
    }

    faceDetectorAvailable = true;

    // load forest for face pose estimation
    if (!estimator.loadForest(poseEstimatorPath.c_str(), 10)) {
        std::cerr << "ERROR! Unable to load forest files" << std::endl;
        return;
    }

    poseEstimatorAvailable = true;
}

// ---------- public member functions ----------

vector<Face> Preprocessor::preprocess(vector<Image4D>& images)
{
    for (auto& image : images)
        segment(image);

    return cropFaces(images);
}


void Preprocessor::segment(std::vector<Image4D>& images)
{
    for (auto& image : images)
        segment(image);

    return;
}


void Preprocessor::segment(Image4D& image4d)
{
    cv::Rect boundingBox;
    if (!detectForegroundFace(image4d, boundingBox))
        removeBackgroundFixed(image4d, FIXED_THRESHOLD);
    else
        removeBackgroundDynamic(image4d, boundingBox);

    return;
}

vector<Face> Preprocessor::cropFaces(vector<Image4D>& images)
{
    const auto SIZE = images.size();

    vector<Face> croppedFaces;
    croppedFaces.reserve(SIZE);

    for (auto& face : images) {
        Vec3f position, eulerAngles;
        auto area = face.getArea();
        //cv::imshow("Non cropped face", face.depthMap);
        //cv::waitKey(0);
        auto cropped = cropFace(face, position, eulerAngles);
        //cv::imshow("Cropped face", face.depthMap);
        //cv::waitKey(0);
        if (!cropped || face.getArea() != area) // keep only images where a face has been detected and cropped
            croppedFaces.emplace_back(face, position, eulerAngles);
    }

    return croppedFaces;
}

bool Preprocessor::cropFace(Image4D& image4d, Vec3f& position, Vec3f& eulerAngles) const
{
    removeOutliers(image4d);

    if (!estimateFacePose(image4d, position, eulerAngles))
        return false;

    cv::Mat normalized;
    cv::normalize(image4d.depthMap, normalized, 0, 255, CV_MINMAX, CV_8U);
    cv::imshow("Depth map", normalized);
    std::cout << eulerAngles << std::endl;
    cv::waitKey(0);

    const int NONZERO_PXL = 5;

    auto yTop = getFirstNonempty<uint16_t>(image4d.depthMap, NONZERO_PXL, ScanOrder::top_down);

    if (std::abs(eulerAngles[0]) > 35)  // TODO: is this if really necessary?
        eulerAngles[0] = 0;

    // necessary corrections to take into account head rotations
    const float BETA  = (eulerAngles[0] > 0) ? 20/8.f : 0.f;
    const float GAMMA = 5/8;
    const float DELTA = 1.1f;
    const float PHI   = 1.f;

    yTop += BETA*eulerAngles[0] + GAMMA*eulerAngles[2];
    if (yTop < 0)
        yTop = 0;

    int rotationFactor = DELTA*std::abs(eulerAngles[0]);
    int distanceFactor = 130 / (position[2] / 1000.f);
    int yBase = yTop + distanceFactor - rotationFactor;
    if (yBase > image4d.getHeight())
        yBase = image4d.getHeight();

    cv::Rect scanROI(0, yTop, image4d.getWidth(), (yBase - yTop) / 2);
    auto roiMat = image4d.depthMap(scanROI);
    auto xTop  = getFirstNonempty<uint16_t>(roiMat, NONZERO_PXL, ScanOrder::left_to_right);
    auto xBase = getFirstNonempty<uint16_t>(roiMat, NONZERO_PXL, ScanOrder::right_to_left);

    // TODO: use a sigmoidal function to minimize lateral cropping for small
    //       values of eulerAngles[1] (but where should it be centered?, in 15?)
    if (eulerAngles[1] > 0)
        xBase -= PHI * std::abs(eulerAngles[1]) - 10;
    else
        xTop  += PHI * std::abs(eulerAngles[1]); // aumentare xTop


    cv::Rect faceROI (xTop, yTop, xBase - xTop, yBase - yTop);
    image4d.crop(faceROI);
    return true;
}

// ---------- private member functions ----------
bool Preprocessor::detectForegroundFace(const Image4D &face, cv::Rect &boundingBox)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(face.image, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(70, 70));

    if (faces.empty())
        return false;

    // take face in foregound (the one with bigger bounding box)
    boundingBox = *std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect &r1, const cv::Rect &r2) { return r1.area() < r2.area(); });

    return true;
}

void Preprocessor::removeBackgroundDynamic(Image4D &face, const cv::Rect &boundingBox) const
{
    // take non-nan, non-zero points
    vector<float> depth;
    auto lambda = [&depth](int x, int y, const uint16_t &dpt) {
        if (std::isnormal(dpt)) // if is not NaN, 0 or INF
            depth.push_back(dpt);
    };

    face.depthForEach<uint16_t>(lambda, boundingBox);

    // clustering
    vector<int> bestLabels;
    vector<float> centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.size() != 2) {
        std::cout << "Clustering on depth map for background removal failed!" << std::endl;
        return;
    }

    // compute threshold based on clustering
    const int FACE_CLUSTER = centers[0] < centers[1] ? 0 : 1;
    float threshold = centers[FACE_CLUSTER] * 1.2f;

    const int MIN_X = boundingBox.x - boundingBox.width;
    const int MAX_X = boundingBox.x + 2 * boundingBox.width;

    face.depthMap.forEach<uint16_t>([&](uint16_t &p, const int *pos) {
        if (float(p) > threshold || std::isnan(p) || pos[1] < MIN_X || pos[1] > MAX_X)
            p = 0;
    });

    return;
}

void Preprocessor::removeBackgroundFixed(Image4D &face, uint16_t threshold) const
{
    face.depthMap.forEach<uint16_t>([threshold](uint16_t &p, const int *pos) {
        if (p > threshold || std::isnan(p))
            p = 0;
    });

    return;
}

void Preprocessor::removeOutliers(Image4D &image4d) const
{
    // TODO: a full resolution booleanDepthMap is probably too much
    //       maybe the same result is achievable with a sampling of a pixel every 4 or 8
    cv::Mat booleanDepthMap(image4d.getHeight(), image4d.getWidth(), CV_8U);
    cv::Mat labels, stats, centroids;
    auto boolIter = booleanDepthMap.begin<bool>();
    for (auto iter = image4d.depthMap.begin<uint16_t>(); iter < image4d.depthMap.end<uint16_t>(); ++iter, ++boolIter) {
        *boolIter = *iter != 0;
    }

    int numOfComponents = cv::connectedComponentsWithStats(booleanDepthMap, labels, stats, centroids, 4);
    int index = 1;
    int maxArea = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < numOfComponents; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > maxArea) {
            maxArea = area;
            index = i;
        }
    }

    int x = stats.at<int>(index, cv::CC_STAT_LEFT);
    int y = stats.at<int>(index, cv::CC_STAT_TOP);
    int width  = stats.at<int>(index, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(index, cv::CC_STAT_HEIGHT);
    cv::Rect roi(x, y, width, height);

    image4d.depthMap.forEach<uint16_t>([&](uint16_t &depth, const int *pos) {
        if (!roi.contains(cv::Point(pos[1], pos[0])))
            depth = 0;
    });
}



bool Preprocessor::estimateFacePose(const Image4D& image4d, cv::Vec3f& position, cv::Vec3f& eulerAngles) const
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    cv::Mat img3D = image4d.get3DImage();

    vector<cv::Vec<float, POSE_SIZE>> means; // outputs, POSE_SIZE defined in CRTree.h
    vector<vector<Vote>> clusters; // full clusters of votes
    vector<Vote> votes; // all votes returned by the forest
    int stride = 10;
    float maxVariance = 800;
    float probTH = 1.0;
    float largerRadiusRatio = 1.5;
    float smallerRadiusRatio = 5.0;
    bool verbose = false;
    int threshold = 500;

    estimator.estimate(img3D, means, clusters, votes, stride, maxVariance,
        probTH, largerRadiusRatio, smallerRadiusRatio, verbose, threshold);

    if (means.empty())
        return false;

    auto &pose = means[0];

    position = { -pose[1] + image4d.getHeight() / 2,
                  pose[0] + image4d.getWidth()  / 2,
                  pose[2] };

    eulerAngles = { pose[3], pose[4], pose[5] };

    return true;
}



// -----------------------------------------------
// ----------- Non member functions --------------
// -----------------------------------------------

template<typename T>
int getFirstNonempty(cv::Mat matrix, int minNonemptySquares, ScanOrder scanOrder)
{
    auto scanByRow  = (scanOrder == ScanOrder::bottom_up || scanOrder == ScanOrder::top_down);
    auto increasing = (scanOrder == ScanOrder::top_down  || scanOrder == ScanOrder::left_to_right);

    int i, j;
    auto &u = scanByRow ? i : j;
    auto &v = scanByRow ? j : i;
    const auto firstDim  = scanByRow ? matrix.rows : matrix.cols;
    const auto secondDim = scanByRow ? matrix.cols : matrix.rows;

    for (auto k = 0; k < firstDim; ++k) { // look for first non-empty row
        int nonzeroSquares = 0;
        i = increasing ? k : firstDim - k - 1;  // TODO: double check this
        for (j = 0; j < secondDim; ++j) {
            if (matrix.at<T>(u, v) != 0)
                ++nonzeroSquares;
        }
        if (nonzeroSquares >= minNonemptySquares) {
            return i;
        }
    }
    return -1;
}

} // nemaspace face
