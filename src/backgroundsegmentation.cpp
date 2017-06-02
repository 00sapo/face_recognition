#include "backgroundsegmentation.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "face.h"
#include "singletonsettings.h"

using std::vector;
using std::string;

const string BackgroundSegmentation::FACE_DETECTOR_PATH = "../haarcascade_frontalface_default.xml";
const string BackgroundSegmentation::POSE_ESTIMATOR_PATH = "../trees/";

BackgroundSegmentation::BackgroundSegmentation() : BackgroundSegmentation(FACE_DETECTOR_PATH, POSE_ESTIMATOR_PATH) { }

BackgroundSegmentation::BackgroundSegmentation(const string& faceDetectorPath, const string& poseEstimatorPath)
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

bool BackgroundSegmentation::detectForegroundFace(const Face& face, cv::Rect& detectedFace)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std:: endl;
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


void BackgroundSegmentation::removeBackground(std::vector<Face>& faces) const
{
    for (auto& face : faces)
        removeBackground(face);
}


bool BackgroundSegmentation::removeBackground(Face& face) const
{
    // build nan mask to skip nans
    vector<float> depth;
    Mat nanMask = Mat::zeros(face.getHeight(), face.getWidth(), CV_8U); // true if nan

    const auto HEIGHT = face.getHeight();
    const auto WIDTH  = face.getWidth();

    for (uint x = 0; x < HEIGHT; ++x) {
        for (uint y = 0; y < WIDTH; ++y) {
            float d = face.depthMap.at<float>(x,y);
            if (std::isnan(d))
                nanMask.at<bool>(x,y) = true;
            else
                depth.push_back(d);
        }
    }

    // clustering
    vector<int>   bestLabels;
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
            if (!nanMask.at<bool>(x,y)) {
                if (*iter != FACE_CLUSTER) {
                    face.depthMap.at<float>(x,y) = 0;
                    face.image.at<uchar>(x,y) = 0;
                }
                ++iter;
            }
            else {
                face.depthMap.at<float>(x,y) = 0;
                face.image.at<uchar>(x,y) = 0;
            }
        }
    }

    return true;
}

bool BackgroundSegmentation::estimateFacePose(const Face& face)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    SingletonSettings& settings = SingletonSettings::getInstance();
    cv::Mat img3D = face.get3DImage(settings.getK());

    cv::imshow("IMage3D", img3D);
    cv::waitKey(0);

    vector<cv::Vec<float, POSE_SIZE>> means; // outputs
    vector<vector<Vote>> clusters;           // full clusters of votes
    vector<Vote> votes;                      // all votes returned by the forest
    int stride = 5;

    estimator.estimate(img3D, means, clusters, votes, stride, 800);

    if (means.empty()) {
        std::cout << "Detection and pose estimation failed!" << std::endl;
        return false;
    }

    for (auto& pose : means) {
        std::cout << "Face detected!" << std::endl;
        std::cout << pose[0] << ", " << pose[1] << ", " << pose[2] << ", "
                  << pose[3] << ", " << pose[4] << ", " << pose[5] << std::endl;
    }

    return true;
}


