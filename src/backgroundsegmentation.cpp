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

bool BackgroundSegmentation::detectForegroundFace(cv::Rect& detectedFace)
{
    // load the pretrained model
    cv::CascadeClassifier classifier("../haarcascade_frontalface_default.xml");
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return false;
    }

    // face detection
    std::vector<cv::Rect> faces;
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
    /**/

    // enlarge a bit the face region
    //    for (auto& bestFace : faces) {
    detectedFace.x -= 20;
    detectedFace.y -= 20;
    detectedFace.height += 40;
    detectedFace.width += 40;
    //    }

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

    std::vector<float> depth;

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
    std::vector<int> bestLabels;
    std::vector<float> centers;

    std::cout << "Clustering..." << std::endl;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
    std::cout << "Done!" << std::endl;

    std::cout << "Size: " << centers.size() << std::endl;
    //std::cout << "Cols: " << centers.cols << std::endl;

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
