#include "backgroundsegmentation.h"

#include <iostream>
#include <opencv2/opencv.hpp>

#include "face.h"

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
    float foregroundFaceDepth = FLT_MAX;
    for (cv::Rect f : faces) {
        for (int x = f.x; x < f.x + f.width; x++) {
            for (int y = f.y; y < f.y + f.height; y++) {
                if (face.cloud->at(x, y).z < foregroundFaceDepth) {
                    foregroundFaceDepth = face.cloud->at(x, y).z;
                    detectedFace = f;
                }
            }
        }
    }

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


void BackgroundSegmentation::removeBackground(Face& face) const
{

    std::cout << "Building depth map..." << std::endl;

    vector<float> depth;

    auto it = face.cloud->begin();
    while (it != face.cloud->end()) {
        if (std::isnan((*it).z)) {
            //            depth.push_back(face.getMinDepth());
            it = face.cloud->erase(it);
        } else {
            depth.push_back((*it).z);
            ++it;
        }
    }
    std::cout << std::endl;

    std::cout << "Done!" << std::endl;

    //    cv::Mat centers = cv::Mat(4, 3, CV_32F).clone();
    vector<int>   bestLabels;
    vector<float> centers;

    std::cout << "Clustering..." << std::endl;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
    std::cout << "Done!" << std::endl;

    std::cout << "Size: " << centers.size() << std::endl;

    const int FACE_CLUSTER = centers.at(0) < centers.at(1) ? 0 : 1;

    std::cout << "Removing background..." << std::endl;
    if (bestLabels.size() != face.cloud->size()) {
        std::cerr << "Can't removing background: labels and cloud have different sizes!!" << std::endl;
        return;
    }
    it = face.cloud->begin();
    for (int label : bestLabels) {
        if (label != FACE_CLUSTER) {
            it = face.cloud->erase(it);
        } else
            ++it;
    }
    std::cout << std::endl
              << "Done!" << std::endl;
}

bool BackgroundSegmentation::estimateFacePose(const Face& face)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    cv::Mat img3D = face.get3DImage();
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


