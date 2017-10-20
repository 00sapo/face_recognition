#include "kmeansbackgroundremover.h"
#include <vector>

using std::vector;

namespace face {
KmeansBackgroundRemover::KmeansBackgroundRemover(uint16_t fixedThreshold)
    : fixedThreshold(fixedThreshold)
{
}

bool KmeansBackgroundRemover::filter(face::Image4DSet& image)
{
    cv::Rect boundingBox;
    // ... detect foreground face...
    if (!detectForegroundFace(image, boundingBox))
        removeBackgroundFixed(image, fixedThreshold);
    else
        removeBackgroundDynamic(image, boundingBox);

    return 1;
}

bool KmeansBackgroundRemover::detectForegroundFace(const Image4DSet& face, cv::Rect& boundingBox)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(*(face.getImage()), faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(70, 70));

    if (faces.empty())
        return false;

    // take face in foregound (the one with bigger bounding box)
    boundingBox = *std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect& r1, const cv::Rect& r2) { return r1.area() < r2.area(); });

    return true;
}

void KmeansBackgroundRemover::removeBackgroundDynamic(Image4DSet& face, const cv::Rect& boundingBox) const
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

    face.getDepthMap()->forEach<uint16_t>([&](uint16_t& p, const int* pos) {
        if (float(p) > threshold || std::isnan(p) || pos[1] < MIN_X || pos[1] > MAX_X)
            p = 0;
    });

    return;
}

void KmeansBackgroundRemover::removeBackgroundFixed(Image4DSet& face, uint16_t threshold) const
{
    face.getDepthMap()->forEach<uint16_t>([threshold](uint16_t& p, const int* pos) {
        if (p > threshold || std::isnan(p))
            p = 0;
    });

    return;
}
}
