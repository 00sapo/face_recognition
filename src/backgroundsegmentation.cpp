#include "backgroundsegmentation.h"

#include <iostream>
#include <math.h>
#include <opencv2/objdetect.hpp>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"
#include "face.h"
#include "pointprojector.h"

using pcl::PointCloud;
using pcl::PointXYZ;
using std::isnan;

BackgroundSegmentation::BackgroundSegmentation(const Face& face)
    : Kmeans(0, 1)
{
    num_clusters_ = 2;
    setFace(face);
}

bool BackgroundSegmentation::detectFaces(std::vector<cv::Rect>& faces)
{
    // load the pretrained model
    cv::CascadeClassifier classifier("../haarcascade_frontalface_default.xml");
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return false;
    }

    // face detection
    //std::vector<cv::Rect> faces;
    classifier.detectMultiScale(face.image, faces);

    // choose foreground face if more than one detected
    /*
    int foregroundFaceindex = -1;
    float foregroundFaceDepth = std::numeric_limits<float>::max();
    for (int i = 0; i < faces.size(); ++i) {

        // estimate depth for candidate face
        float depth = 0;
        for (int j = 0; j < 10; j++) {

        }


    }
    */

    // enlarge a bit the face region
    for (auto& face : faces) {
        face.x -= 20;
        face.y -= 20;
        face.height += 40;
        face.width += 40;
    }

    return !faces.empty();
}

void BackgroundSegmentation::findClusters()
{

    std::cout << "Adding points to KMeans..." << std::endl;

    //for (unsigned int i = 0; i < face.cloud->size(); ++i) {
    //    float value = face.cloud->at(i).z;
    for (auto& point : *face.cloud) {
        float value = point.z;
        if (std::isnan(value)) {
            value = FLT_MIN;
        }
        pcl::Kmeans::Point p = { value };
        this->addDataPoint(p);
    }
    std::cout << "Done!" << std::endl;

    std::cout << "Clustering..." << std::endl;
    this->kMeans();
    std::cout << "Done!" << std::endl;
}

void BackgroundSegmentation::filter()
{
    Point min = { FLT_MAX };
    uint clusterId = 0;
    for (uint i = 0; i < num_clusters_; i++) {
        if (centroids_[i][0] < min[0]) {
            clusterId = i;
            min = centroids_[i];
        }
    }

    const uint WIDTH = face.getWidth();
    const uint HEIGHT = face.getHeight();

    float nan = std::numeric_limits<float>::quiet_NaN();

    cv::Mat filteredImage = cv::Mat::zeros(HEIGHT, WIDTH, CV_8U);
    PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>(WIDTH, HEIGHT));

    for (ulong i = 0; i < face.cloud->size(); ++i) {
        uint x = i / WIDTH;
        uint y = i % WIDTH;
        if (points_to_clusters_[i] == clusterId) {
            const auto& point = face.cloud->at(i);
            filteredCloud->at(i) = point;
            if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                filteredImage.at<uchar>(x, y) = face.image.at<uchar>(x, y);
            }
        } else {
            filteredCloud->at(y, x) = { nan, nan, nan };
        }
    }

    face.cloud = filteredCloud;
    face.image = filteredImage;
}

void BackgroundSegmentation::filterBackground()
{
    std::cout << "Looking for threshold..." << std::endl;

    findClusters();
    std::cout << "Done!" << std::endl;

    std::cout << "Removing background..." << std::endl;
    filter();
    std::cout << "Done!" << std::endl;
}

void BackgroundSegmentation::filterBackground(std::vector<Face>& faces)
{

    for (auto& face : faces) {
        setFace(face);
        filterBackground();
    }
}

Face BackgroundSegmentation::getFace() const
{
    return face;
}

void BackgroundSegmentation::setFace(const Face& value)
{
    face = value;
    num_points_ = face.cloud->size();
    points_to_clusters_ = PointsToClusters(num_points_, 0);
}

void BackgroundSegmentation::estimateFacePose()
{
    CRForestEstimator estimator;
    if (!estimator.loadForest("../trees/", 10)) {
        std::cerr << "Can't find forest files" << std::endl;
    }

    cv::Mat img3D = face.get3DImage();
    std::vector<cv::Vec<float, POSE_SIZE>> means; //outputs
    std::vector<std::vector<Vote>> clusters; //full clusters of votes
    std::vector<Vote> votes; //all votes returned by the forest
    int stride = 5;

    estimator.estimate(img3D, means, clusters, votes, stride, 800);

    if (means.empty())
        std::cout << "Detection and pose estimation failed!" << std::endl;

    for (auto& pose : means) {
        std::cout << "Face detected!" << std::endl;
        std::cout << pose[0] << ", " << pose[1] << ", " << pose[2] << ", "
                  << pose[3] << ", " << pose[4] << ", " << pose[5] << std::endl;
    }
}

void BackgroundSegmentation::removeBackground(Face& face)
{

    std::cout << "Building depth map..." << std::endl;

    //    cv::Mat depth = cv::Mat(face.cloud->size(), 3, CV_32FC1).clone();
    //    //cv::Mat bestLabels(face.cloud->size(), 1, CV_32S);
    //    const float THRESHOLD = (face.getMinDepth() + face.getMaxDepth()) / 2;

    //    for (uint i = 0; i < face.cloud->size(); ++i) {
    //        const auto& point = face.cloud->at(i);
    //        //if (pointDepth != NAN)

    //        depth.at<float>(i, 0) = point.x; //{face.cloud->at(i).x,face.cloud->at(i).y,face.cloud->at(i).z};
    //        depth.at<float>(i, 1) = point.y;
    //        depth.at<float>(i, 2) = point.z;
    //        //bestLabels.at<int>(i) = pointDepth < THRESHOLD ? 0 : 1;
    //    }
    //cv::Mat depth = face.get3DImage();

    cv::Mat depth = cv::Mat(face.cloud->size(), 1, CV_32F);
    for (uint i = 0; i < face.cloud->size(); ++i) {
        const auto& point = face.cloud->at(i);
        if (std::isnan(point.z))
            depth.at<float>(i, 0) = 0;
        else
            depth.at<float>(i, 0) = point.z;

        std::cout << depth.at<float>(i, 0) << "; ";
    }
    std::cout << std::endl;

    std::cout << "Done!" << std::endl;

    //    cv::Mat centers = cv::Mat(4, 3, CV_32F).clone();
    std::vector<int> bestLabels;
    cv::Mat centers = cv::Mat(1, 2, CV_32F);

    std::cout << "Clustering..." << std::endl;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
    std::cout << "Done!" << std::endl;

    std::cout << "Size: " << centers.size() << std::endl;
    //std::cout << "Cols: " << centers.cols << std::endl;

    const int FACE_CLUSTER = 1; //centers.row(0) < centers.row(1) ? 0 : 1;

    std::cout << "Removing background..." << std::endl;
    for (uint i = 0; i < face.cloud->size(); ++i) {
        if (bestLabels.at(i) != FACE_CLUSTER) {
            std::cout << bestLabels.at(i) << std::endl;
            face.cloud->erase(face.cloud->begin() + i);
        }
    }
    std::cout << "Done!" << std::endl;
}
