#include "facesegmenter.h"

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "face.h"
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

bool FaceSegmenter::detectForegroundFace(const Face& face, const Size& outputSize, cv::Rect& detectedRegion)
{

    assert (outputSize.width <= face.getWidth() && outputSize.height <= face.getHeight()
            && "Output region can't be bigger than input face image!");

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

    if (detectedRegion.width >= outputSize.width && detectedRegion.height >= outputSize.height)
        return !faces.empty();

    // enlarge region to have desired aspect ratio
    int widthEnlarge  = outputSize.width  - detectedRegion.width;
    int heightEnlarge = outputSize.height - detectedRegion.height;

    detectedRegion.x -= widthEnlarge/2;
    if (detectedRegion.x < 0) {
        widthEnlarge += -detectedRegion.x;
        detectedRegion.x = 0;
    }
    detectedRegion.width += widthEnlarge;
    detectedRegion.width  = detectedRegion.width + detectedRegion.x > face.getWidth() ?
                face.getWidth() - detectedRegion.x :  detectedRegion.width;

    detectedRegion.y -= heightEnlarge/2;
    if (detectedRegion.y < 0) {
        heightEnlarge += -detectedRegion.y;
        detectedRegion.y = 0;
    }
    detectedRegion.height += heightEnlarge;
    detectedRegion.height  = detectedRegion.height + detectedRegion.y > face.getHeight() ?
                face.getHeight() - detectedRegion.y :  detectedRegion.height;

    return !faces.empty();
}

void FaceSegmenter::removeBackground(std::vector<Face>& faces) const
{
    for (auto& face : faces)
        removeBackground(face);
}

bool FaceSegmenter::removeBackground(Face& face) const
{
    // build nan mask to skip nans
    vector<float> depth;
    Mat nanMask = Mat::zeros(face.getHeight(), face.getWidth(), CV_8U); // true if nan

    const auto HEIGHT = face.getHeight();
    const auto WIDTH = face.getWidth();

    for (uint x = 0; x < HEIGHT; ++x) {
        for (uint y = 0; y < WIDTH; ++y) {
            float d = face.depthMap.at<uint16_t>(x, y);
            if (std::isnan(d))
                nanMask.at<bool>(x, y) = true;
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
    const int FACE_CLUSTER = centers.at(0) < centers.at(1) ? 0 : 1;
    auto iter = bestLabels.begin();

    for (uint x = 0; x < HEIGHT; ++x) {
        for (uint y = 0; y < WIDTH; ++y) {
            if (!nanMask.at<bool>(x, y)) {
                if (*iter != FACE_CLUSTER) {
                    face.depthMap.at<uint16_t>(x, y) = 0;
                    face.image.at<uchar>(x, y) = 0;
                }
                ++iter;
            } else {
                face.depthMap.at<uint16_t>(x, y) = 0;
                face.image.at<uchar>(x, y) = 0;
            }
        }
    }

    return true;
}
