#include "preprocessor.h"

#include <cmath>
#include <thread>

#include "face.h"
#include "image4d.h"

using cv::Vec3f;
using std::string;
using std::vector;

namespace face {

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
    segment(images);
    auto faces = cropFaces(images);

    return faces;
}

void Preprocessor::segment(std::vector<Image4D>& images)
{

    std::cout << "segmenting..." << std::endl;
    // for each image...
    for (auto& image : images) {
        std::cout << "segmenting image " << image.name << std::endl;
        segment(image);
    }

    return;
}

vector<Face> Preprocessor::cropFaces(vector<Image4D>& images)
{
    const auto SIZE = images.size();

    std::cout << "cropping faces..." << std::endl;
    vector<Face> croppedFaces;
    croppedFaces.reserve(SIZE);

    for (auto& face : images) {
        std::cout << "cropping faces " << face.name << std::endl;
        Vec3f position, eulerAngles;
        cropFace(face, position, eulerAngles);
        croppedFaces.emplace_back(face, position, eulerAngles);
    }

    return croppedFaces;
}

// ---------- private member functions ----------

void Preprocessor::segment(Image4D& image4d)
{
    cv::Rect boundingBox;
    // ... detect foreground face...
    if (!detectForegroundFace(image4d, boundingBox)) {
        //        std::cout << "No face detected!"
        //                  << " Applying fixed threshold." << std::endl;
        removeBackgroundFixed(image4d, 1600);
    } else {

        removeBackgroundDynamic(image4d, boundingBox);
    }

    return;
}

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
        if (!std::isnan(dpt) && dpt != 0)
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

void Preprocessor::removeOutliers(Image4D& image4d, float threshold) const
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

    //image4d.crop(roi);

    image4d.depthMap.forEach<uint16_t>([&](uint16_t& depth, const int* pos) {
        if (!roi.contains(cv::Point(pos[1], pos[0])))
            depth = 0;
    });
}

bool Preprocessor::cropFace(Image4D& image4d, Vec3f& position, Vec3f& eulerAngles) const
{
    //cv::imshow("Pre", image4d.depthMap);
    //cv::waitKey(0);
    removeOutliers(image4d);
    //cv::imshow("Post", image4d.depthMap);
    //cv::waitKey(0);

    if (!estimateFacePose(image4d, position, eulerAngles)) {
        return false;
    }

    //    std::cout << "Face detected!" << std::endl;
    //    std::cout << "Position: " << position[0] << "," << position[1] << "," << position[2] << std::endl;
    //    std::cout << "Euler angles: " << eulerAngles[0] << "," << eulerAngles[1] << "," << eulerAngles[2] << std::endl;

    const auto HEIGHT = image4d.getHeight();
    const auto WIDTH = image4d.getWidth();
    const int NONZERO_PXL_THRESHOLD = 5;

    int yTop = 0;
    for (std::size_t i = 0; i < HEIGHT; ++i) { // look for first non-empty row
        int nonzeroPixels = 0;
        for (std::size_t j = 0; j < WIDTH; ++j) {
            if (image4d.depthMap.at<uint16_t>(i, j) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            yTop = i;
            break;
        }
    }

    if (std::abs(eulerAngles[0]) > 35)
        eulerAngles[0] = 0;

    // necessary corrections to take into account head rotations
    yTop += 10 / 8 * eulerAngles[0] + 5 / 8 * eulerAngles[2];
    if (yTop < 0)
        yTop = 0;
    int yBase = yTop + (145 / (position[2] / 1000.f));
    if (yBase > HEIGHT)
        yBase = HEIGHT;
    cv::Rect faceROI(0, yTop, WIDTH, yBase - yTop);

    const int MAX_Y = faceROI.y + faceROI.height - 30; // stay 30px higher to avoid shoulders

    int xTop = 0;
    for (int i = 0; i < WIDTH; ++i) { // look for first non-empty column from left
        int nonzeroPixels = 0;
        for (int j = faceROI.y; j < MAX_Y; ++j) {
            if (image4d.depthMap.at<uint16_t>(j, i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xTop = i;
            break;
        }
    }

    int xBase = 0;
    for (int i = WIDTH - 1; i >= 0; --i) { // look for last non-empty column from right
        int nonzeroPixels = 0;
        for (int j = faceROI.y; j < MAX_Y; ++j) {
            if (image4d.depthMap.at<uint16_t>(j, i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xBase = i;
            break;
        }
    }

    faceROI.x = xTop;
    faceROI.width = xBase - xTop;

    image4d.crop(faceROI);
    return true;
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

    if (means.empty()) {
        //        std::cout << "Detection and pose estimation failed!" << std::endl;
        return false;
    }

    auto& pose = means[0];

    position = { -pose[1] + image4d.getHeight() / 2,
        pose[0] + image4d.getWidth() / 2,
        pose[2] };

    //    std::cout << "Angles: " << pose[3] << "," << pose[4] << "," << pose[5] << std::endl;
    eulerAngles = { pose[3], pose[4], pose[5] };

    return true;
}

} // nemaspace face
