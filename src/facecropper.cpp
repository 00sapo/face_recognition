#include "facecropper.h"
#include <settings.h>
#include <vector>

using std::vector;

namespace face {
FaceCropper::FaceCropper()
{
    // load forest for face pose estimation
    const char* estimatorPath = Settings::getInstance().getPoseEstimatorPath().c_str();
    if (!estimator.loadForest(estimatorPath, 10)) {
        std::cerr << "ERROR! Unable to load forest files" << std::endl;
        return;
    }

    poseEstimatorAvailable = true;
}

void FaceCropper::removeOutliers() const
{
    // TODO: a full resolution booleanDepthMap is probably too much
    //       maybe the same result is achievable with a sampling of a pixel every 4 or 8
    cv::Mat booleanDepthMap(image4d->getHeight(), image4d->getWidth(), CV_8U);
    cv::Mat labels, stats, centroids;
    auto boolIter = booleanDepthMap.begin<bool>();
    for (auto iter = image4d->getDepthMap().begin<uint16_t>(); iter < image4d->getDepthMap().end<uint16_t>(); ++iter, ++boolIter) {
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

    image4d->getDepthMap().forEach<uint16_t>([&](uint16_t& depth, const int* pos) {
        if (!roi.contains(cv::Point(pos[1], pos[0])))
            depth = 0;
    });
}

void FaceCropper::filterImage4DComponent(Image4DComponent* image4d)
{
    FaceCropper cropper;
    cropper.setImage4DComponent(image4d);
    cropper.filter();
}

bool FaceCropper::filter()
{

    std::cout << "Cropping faces..." << std::endl;
    Image4DComponent* backupImage = image4d;
    if (image4d->isLeaf()) {
        crop();
    } else {
        for (Image4DComponent* img4d : *backupImage) {
            setImage4DComponent(img4d);
            filter();
        }
    }
    setImage4DComponent(backupImage);
    return true;
}

bool FaceCropper::crop()
{
    removeOutliers();

    if (!estimateFacePose()) {
        return false;
    }

    const auto HEIGHT = image4d->getHeight();
    const auto WIDTH = image4d->getWidth();
    const int NONZERO_PXL_THRESHOLD = 5;

    int yTop = 0;
    for (std::size_t i = 0; i < HEIGHT; ++i) { // look for first non-empty row
        int nonzeroPixels = 0;
        for (std::size_t j = 0; j < WIDTH; ++j) {
            if (image4d->getDepthMap().at<uint16_t>(i, j) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            yTop = i;
            break;
        }
    }

    if (std::abs(image4d->getEulerAngles()[0]) > 35)
        image4d->getEulerAngles()[0] = 0;

    // necessary corrections to take into account head rotations
    yTop += 10 / 8 * image4d->getEulerAngles()[0] + 5 / 8 * image4d->getEulerAngles()[2];
    if (yTop < 0)
        yTop = 0;
    int yBase = yTop + (145 / (image4d->getPosition()[2] / 1000.f));
    if (yBase > HEIGHT)
        yBase = HEIGHT;
    cv::Rect faceROI(0, yTop, WIDTH, yBase - yTop);

    const int MAX_Y = faceROI.y + faceROI.height - 30; // stay 30px higher to avoid shoulders

    int xTop = 0;
    for (size_t i = 0; i < WIDTH; ++i) { // look for first non-empty column from left
        int nonzeroPixels = 0;
        for (int j = faceROI.y; j < MAX_Y; ++j) {
            if (image4d->getDepthMap().at<uint16_t>(j, i) != 0)
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
            if (image4d->getDepthMap().at<uint16_t>(j, i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xBase = i;
            break;
        }
    }

    faceROI.x = xTop;
    faceROI.width = xBase - xTop;

    image4d->crop(faceROI);
    return true;
}

bool FaceCropper::estimateFacePose() const
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    cv::Mat img3D = image4d->get3DImage();

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

    image4d->setPosition({ -pose[1] + image4d->getHeight() / 2,
        pose[0] + image4d->getWidth() / 2,
        pose[2] });

    image4d->setEulerAngles({ pose[3], pose[4], pose[5] });

    return true;
}

CRForestEstimator FaceCropper::getEstimator() const
{
    return estimator;
}

void FaceCropper::setEstimator(const CRForestEstimator& value)
{
    estimator = value;
    poseEstimatorAvailable = true;
}

bool FaceCropper::isPoseEstimatorAvailable() const
{
    return poseEstimatorAvailable;
}

Image4DComponent* FaceCropper::getImage4DComponent() const
{
    return image4d;
}

void FaceCropper::setImage4DComponent(Image4DComponent* value)
{
    image4d = value;
}
}
