#include "facesegmenter.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "image4d.h"
#include "singletonsettings.h"

using std::vector;
using std::string;

const string FaceSegmenter::FACE_DETECTOR_PATH = "../haarcascade_frontalface_default.xml";

FaceSegmenter::FaceSegmenter()
    : FaceSegmenter(FACE_DETECTOR_PATH)
{
}

FaceSegmenter::FaceSegmenter(const string& faceDetectorPath)
{
    // load the pretrained face detection model
    classifier = cv::CascadeClassifier(faceDetectorPath);
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return;
    }

    faceDetectorAvailable = true;
}

bool FaceSegmenter::detectForegroundFace(const Image4D& face, cv::Rect& detectedRegion)
{

    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(face.image, faces);

    // TODO: choose foreground face if more than one detected
    /*
    float foregroundFaceDepth = FLT_MAX;
    cv::Rect detectedFace;
    for (cv::Rect f : faces) {
        float averageDepth = 0;
        int count = 0;
        face.depthForEach([&averageDepth, &count] (int x, int y, const float& depth) mutable {
            if (depth != std::numeric_limits<float>::quiet_NaN()) {
                count++;
                averageDepth += depth;
                std::cout << depth << std::endl;
            }
        }, f);
        averageDepth /= count;
        if (averageDepth < foregroundFaceDepth) {
            foregroundFaceDepth = averageDepth;
            detectedFace = f;
        }
    }
    */

    detectedRegion = faces[0];

    return !faces.empty();
}

void FaceSegmenter::removeBackground(std::vector<Image4D>& faces, const cv::Rect &roi) const
{
    for (auto& face : faces)
        removeBackground(face, roi);
}

bool FaceSegmenter::removeBackground(Image4D& face, const cv::Rect& roi) const
{

    const auto HEIGHT = face.getHeight();
    const auto WIDTH = face.getWidth();

    const uint maxWidth  = roi.width  + roi.x;
    const uint maxHeight = roi.height + roi.y;

    assert (roi.x > 0 && roi.y > 0 && maxWidth <= WIDTH && maxHeight <= HEIGHT
            && "ROI must be included in face.image");

    // build nan mask to skip nans
    vector<float> depth;
    cv::Mat nanMask = cv::Mat::zeros(maxHeight, maxWidth, CV_8U); // true if nan

    for (uint i = roi.y; i < maxHeight; ++i) {
        for (uint j = roi.x; j < maxWidth; ++j) {
            float d = face.depthMap.at<uint16_t>(i, j);
            if (std::isnan(d))
                nanMask.at<bool>(i - roi.y, j - roi.x) = true;
            else
                depth.push_back(d);
        }
    }

    // clustering
    vector<int> bestLabels;
    vector<float> centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.size() != 2) {
        std::cout << "Clustering on depth map for background removal failed!" << std::endl;
        return false;
    }

    // remove background
    const int FACE_CLUSTER = centers[0] < centers[1] ? 0 : 1;

    float threshold = centers[FACE_CLUSTER];
    threshold *= 1.2f;

    std::cout << "Threshold: " << threshold << std::endl;

    imshow("Depth map", face.depthMap);
    cv::waitKey(0);

    // using opencv's parallel foreach to take advantage of multithreading
    face.depthMap.forEach<uint16_t>([threshold](uint16_t &p, const int *position) -> void {
        float depth = p;
        if (depth > threshold || depth == std::numeric_limits<float>::quiet_NaN()) {
            p = 0;
        }
    });

    imshow("Depth map", face.depthMap);
    cv::waitKey(0);

    return true;
}
