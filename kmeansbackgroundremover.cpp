#include "kmeansbackgroundremover.h"
#include <settings.h>
#include <vector>

using std::vector;

namespace face {
KmeansBackgroundRemover::KmeansBackgroundRemover(uint16_t fixedThreshold)
    : fixedThreshold(fixedThreshold)
{
    // load the pretrained face detection model
    classifier = cv::CascadeClassifier(Settings::getInstance().getFaceDetectorPath());
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return;
    }

    faceDetectorAvailable = true;
}

bool KmeansBackgroundRemover::filter()
{
    cv::Rect boundingBox;
    // ... detect foreground face...
    if (!detectForegroundFace(boundingBox))
        removeBackgroundFixed();
    else
        removeBackgroundDynamic(boundingBox);

    return 1;
}

Image4DSetComponent* KmeansBackgroundRemover::getImage4d() const
{
    return image4d;
}

void KmeansBackgroundRemover::setImage4d(Image4DSetComponent* value)
{
    image4d = value;
}

uint16_t KmeansBackgroundRemover::getFixedThreshold() const
{
    return fixedThreshold;
}

void KmeansBackgroundRemover::setFixedThreshold(const uint16_t& value)
{
    fixedThreshold = value;
}

cv::CascadeClassifier KmeansBackgroundRemover::getClassifier() const
{
    return classifier;
}

void KmeansBackgroundRemover::setClassifier(const cv::CascadeClassifier& value)
{
    classifier = value;
    faceDetectorAvailable = true;
}

bool KmeansBackgroundRemover::isFaceDetectorAvailable() const
{
    return faceDetectorAvailable;
}

bool KmeansBackgroundRemover::detectForegroundFace(cv::Rect& boundingBox)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(image4d->getImage(), faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(70, 70));

    if (faces.empty())
        return false;

    // take face in foregound (the one with bigger bounding box)
    boundingBox = *std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect& r1, const cv::Rect& r2) { return r1.area() < r2.area(); });

    return true;
}

void KmeansBackgroundRemover::removeBackgroundDynamic(cv::Rect& boundingBox) const
{
    // take non-nan, non-zero points
    vector<float> depth;
    auto lambda = [&depth](int x, int y, const uint16_t& dpt) {
        if (!std::isnan(dpt) && dpt != 0)
            depth.push_back(dpt);
    };

    image4d->depthForEach<uint16_t>(lambda, boundingBox);

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

    image4d->getDepthMap().forEach<uint16_t>([&](uint16_t& p, const int* pos) {
        if (float(p) > threshold || std::isnan(p) || pos[1] < MIN_X || pos[1] > MAX_X)
            p = 0;
    });

    return;
}

void KmeansBackgroundRemover::removeBackgroundFixed() const
{
    uint16_t threshold = fixedThreshold;
    image4d->getDepthMap().forEach<uint16_t>([threshold](uint16_t& p, const int* pos) {
        if (p > threshold || std::isnan(p))
            p = 0;
    });

    return;
}
}
