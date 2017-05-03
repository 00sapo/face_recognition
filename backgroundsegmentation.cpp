#include "backgroundsegmentation.h"

#include <iostream>
#include <math.h>
#include <pointprojector.h>
#include <opencv2/objdetect.hpp>

#include "face.h"
#include "head_pose_estimation/CRForestEstimator.h"

using pcl::PointCloud;
using pcl::PointXYZ;
using std::isnan;

BackgroundSegmentation::BackgroundSegmentation(const Face& face)
    : Kmeans(0, 1)
{
    num_clusters_ = 2;
    setFace(face);
}

bool BackgroundSegmentation::detectFaces(std::vector<cv::Rect> &faces)
{
    cv::CascadeClassifier classifier("../haarcascade_frontalface_default.xml");
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return false;
    }
    classifier.detectMultiScale(face.image, faces);
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

    const uint WIDTH  = face.getWidth();
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
        }
        else {
            filteredCloud->at(y,x) = {nan, nan, nan};
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

void BackgroundSegmentation::cropFace()
{
    CRForestEstimator estimator;
    if(!estimator.loadForest("../trees/", 10)) {
        std::cerr << "Can't find forest files" << std::endl;
    }

    cv::Mat img3D = face.get3DImage();
    std::vector< cv::Vec<float,POSE_SIZE> > means; //outputs
    std::vector< std::vector< Vote > > clusters; //full clusters of votes
    std::vector< Vote > votes; //all votes returned by the forest
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
