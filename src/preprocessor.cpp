#include "preprocessor.h"
#include <thread>

#include <cmath>
#include <preprocessor.h>
#include <thread>

#include "face.h"
#include "image4d.h"
#include "utils.h"

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
template <typename T>
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

void Preprocessor::maskRGBToDepth(Image4D& image)
{
    cv::Mat imgNoBackground, mask;
    image.depthMap.convertTo(mask, CV_8U);
    image.image.copyTo(imgNoBackground, mask);
    image.image = imgNoBackground;
}

vector<Face> Preprocessor::preprocess(vector<Image4D> images)
{
    const auto SIZE = images.size();
    vector<Face> croppedFaces;

    // get number of concurrently executable threads
    const auto numOfThreads = 1;//std::thread::hardware_concurrency();
    vector<std::thread> threads(numOfThreads);

    // equally split number of images to process
    int blockSize = SIZE / numOfThreads;
    if (blockSize < 1)
        blockSize = 1;

    std::mutex cropMutex;
    for (int i = 0; i < numOfThreads && i < SIZE; ++i) {
        int begin = i * blockSize;
        int end = begin + blockSize;
        if (i == numOfThreads - 1)
            end += SIZE % numOfThreads;

        // start a thread executing threadGet function
        threads[i] = std::thread(&Preprocessor::cropFaceThread, this, std::ref(images),
            std::ref(croppedFaces), begin, end, std::ref(cropMutex));
    }

    // wait for threads to end (syncronization)
    for (auto& thread : threads)
        thread.join();

    return croppedFaces;
}

//void Preprocessor::segment(std::vector<Image4D>& images)
//{
//    for (auto& image : images)
//        segment(image);
//
//    return;
//}
//
//void Preprocessor::segment(Image4D& image4d)
//{
//    cv::Rect boundingBox;
//    if (!detectForegroundFace(image4d, boundingBox))
//        removeBackgroundFixed(image4d, FIXED_THRESHOLD);
//    else
//        removeBackgroundDynamic(image4d, boundingBox);
//
//    return;
//}

void Preprocessor::cropFaceThread(const vector<Image4D>& inputFaces, vector<Face>& croppedFaces,
    int begin, int end, std::mutex& cropMutex)
{
    for (auto i = begin; i < end; ++i) {
        const auto& image4D = inputFaces[i];

        auto area = image4D.getArea();
        Face croppedFace;
        auto cropped = cropFace(image4D, croppedFace);

        if (cropped && croppedFace.getArea() != area) { // keep only images where a face has been detected and cropped
            std::lock_guard<std::mutex> lock(cropMutex);
            croppedFaces.push_back(croppedFace);
        }
    }
}

bool Preprocessor::cropFace(const Image4D& image4d, Face& croppedFace)
{
    //removeOutliers(image4d);
    Vec3f position, eulerAngles;
    if (!estimateFacePose(image4d, position, eulerAngles))
        return false;

    const int NONZERO_PXL = 5;

    // necessary corrections to take into account head rotations
    const float BETA = (eulerAngles[0] > 0) ? 15 / 8.f : 0.f;
    const float GAMMA = 5 / 8.f;
    const float DELTA = 1.1f;
    const float PHI = 1.5f;

    auto yTop = getFirstNonempty<uint16_t>(image4d.depthMap, NONZERO_PXL, ScanOrder::top_down);

    if (yTop == -1) {
        std::cout << "WARNING! getFirstNonempty() == -1" << std::endl;
        return false;
    }

    yTop += BETA * eulerAngles[0] + GAMMA * eulerAngles[2];
    if (yTop < 0)
        yTop = 0;

    int rotationFactor = DELTA * std::abs(eulerAngles[0]);
    int distanceFactor = 120 / (position[2] / 1000.f);
    int yBase = yTop + distanceFactor - rotationFactor;
    if (yBase > image4d.getHeight())
        yBase = image4d.getHeight();

    // scan only the upper part of the image to avoid shoulders
    cv::Rect scanROI(0, yTop, image4d.getWidth(), (yBase - yTop) / 2);

    // if looking downwards scan only the lower part of the image
    if (eulerAngles[0] < 0) {
        scanROI.y = yTop + (yBase - yTop) / 2;
        scanROI.height = (yBase - yTop) / 2;
    }

    auto roiMat = image4d.depthMap(scanROI);

    auto xTop = getFirstNonempty<uint16_t>(roiMat, NONZERO_PXL, ScanOrder::left_to_right);
    auto xBase = getFirstNonempty<uint16_t>(roiMat, NONZERO_PXL, ScanOrder::right_to_left);

    if (xTop == -1 || xBase == -1) {
        std::cout << "WARNING! getFirstNonempty() == -1" << std::endl;
        return false;
    }

    // TODO: use a sigmoidal function to minimize lateral cropping for small
    //       values of eulerAngles[1] (but where should it be centered?, in 15?)
    if (eulerAngles[1] > 0)
        xBase -= PHI * std::abs(eulerAngles[1]) - 10;
    else
        xTop += PHI * std::abs(eulerAngles[1]) - 10; // aumentare xTop

    if (xTop < 0 || yTop < 0 || xBase - xTop < 0 || yBase - yTop < 0)
        return false;

    cv::Rect faceROI(xTop, yTop, xBase - xTop, yBase - yTop);
    if (faceROI.height <= 2 && faceROI.width <= 2)
        return false;

    Image4D cropped;
    image4d.crop(faceROI, cropped);

    croppedFace = Face(cropped, position, eulerAngles);

    return true;
}

// ---------- private member functions ----------
bool Preprocessor::detectForegroundFace(const Image4D& face, cv::Rect& boundingBox)
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
        [](const cv::Rect& r1, const cv::Rect& r2) { return r1.area() < r2.area(); });

    return true;
}

void Preprocessor::removeBackgroundDynamic(Image4D& face, const cv::Rect& boundingBox) const
{
    // take non-nan, non-zero points
    vector<float> depth;
    auto lambda = [&depth](int x, int y, const uint16_t& dpt) {
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

    face.depthMap.forEach<uint16_t>([&](uint16_t& p, const int* pos) {
        if (float(p) > threshold || std::isnan(p) || pos[1] < MIN_X || pos[1] > MAX_X)
            p = 0;
    });

    return;
}

void Preprocessor::removeBackgroundFixed(Image4D& face, uint16_t threshold) const
{
    face.depthMap.forEach<uint16_t>([threshold](uint16_t& p, const int* pos) {
        if (p > threshold || std::isnan(p))
            p = 0;
    });

    return;
}

void Preprocessor::removeOutliers(Image4D& image4d)
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
    int width = stats.at<int>(index, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(index, cv::CC_STAT_HEIGHT);
    cv::Rect roi(x, y, width, height);

    image4d.depthMap.forEach<uint16_t>([&](uint16_t& depth, const int* pos) {
        if (!roi.contains(cv::Point(pos[1], pos[0])))
            depth = 0;
    });
}

bool Preprocessor::estimateFacePose(const Image4D& image4d, cv::Vec3f& position, cv::Vec3f& eulerAngles)
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

    auto& pose = means[0];

    position = { -pose[1] + image4d.getHeight() / 2,
        pose[0] + image4d.getWidth() / 2,
        pose[2] };

    eulerAngles = { pose[3], pose[4], pose[5] };

    return true;
}

// -----------------------------------------------
// ----------- Non member functions --------------
// -----------------------------------------------

template <typename T>
int getFirstNonempty(cv::Mat matrix, int minNonemptySquares, ScanOrder scanOrder)
{
    auto scanByRow = (scanOrder == ScanOrder::bottom_up || scanOrder == ScanOrder::top_down);
    auto increasing = (scanOrder == ScanOrder::top_down || scanOrder == ScanOrder::left_to_right);

    int i, j;
    auto& u = scanByRow ? i : j;
    auto& v = scanByRow ? j : i;
    const auto firstDim = scanByRow ? matrix.rows : matrix.cols;
    const auto secondDim = scanByRow ? matrix.cols : matrix.rows;

    for (auto k = 0; k < firstDim; ++k) {
        int nonzeroSquares = 0;
        i = increasing ? k : firstDim - k - 1;
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

} // namespace face
