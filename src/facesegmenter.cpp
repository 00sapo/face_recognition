#include "facesegmenter.h"

#include <iostream>
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

bool FaceSegmenter::detectForegroundFace(const Face& face, cv::Rect& detectedFace)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(face.image, faces);

    // choose foreground face if more than one detected
    //float foregroundFaceDepth = FLT_MAX;
    //for (cv::Rect f : faces) {
    //    face.depthForEach([detectedFace, foregroundFaceDepth, f] (int x, int y, const float& depth) mutable {
    //        if (depth < foregroundFaceDepth) {
    //            foregroundFaceDepth = depth;
    //            detectedFace = f;
    //        }
    //    }, f);
    //}

    detectedFace = faces[0];

    // enlarge a bit the face region
    detectedFace.x -= 20;
    detectedFace.y -= 20;
    detectedFace.height += 40;
    detectedFace.width += 40;

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
            uint16_t d = face.depthMap.at<uint16_t>(x, y);
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

    for (uint x = 0; x < HEIGHT; ++x) {
        for (uint y = 0; y < WIDTH; ++y) {
            /*
            if (!nanMask.at<bool>(x, y)) {
                if (*iter != FACE_CLUSTER) {
                    face.depthMap.at<float>(x, y) = 0;
                    face.image.at<uchar>(x, y) = 0;
                }
                ++iter;
            } else {
                face.depthMap.at<float>(x, y) = 0;
                face.image.at<uchar>(x, y) = 0;
            }
            */
            if (face.depthMap.at<uint16_t>(x, y) > 2) {
                face.depthMap.at<uint16_t>(x, y) = 0;
                face.image.at<uchar>(x, y) = 0;
            }
        }
    }

    return true;
}
