#include "backgroundsegmentation.h"

#include <iostream>

#include "face.h"

using pcl::PointCloud;
using pcl::PointXYZ;

BackgroundSegmentation::BackgroundSegmentation()
    : Kmeans(0, 1)
{
    num_clusters_ = 2;
}

float BackgroundSegmentation::findThreshold(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

    std::cout << "Adding points to KMeans..." << std::endl;
    num_points_ = cloud->size();
    points_to_clusters_ = PointsToClusters(num_points_, 0);

    for (unsigned int i = 0; i < cloud->size(); ++i) {
        pcl::Kmeans::Point p = { cloud->at(i).z };
        this->addDataPoint(p);
    }
    std::cout << "Done!" << std::endl;

    std::cout << "Clustering..." << std::endl;
    this->kMeans();
    std::cout << "Done!" << std::endl;

    /* TODO verificare se bisogna prendere l'indice 0 o 1*/
    std::cout << "Set points..." << std::endl;
    pcl::Kmeans::SetPoints background = this->clusters_to_points_[1];
    std::cout << "Done!" << std::endl;

    std::cout << "Computing min thresh..." << std::endl;
    float minimum = std::numeric_limits<float>::max();
    for (auto& t : background) {
        if (cloud->at(t).z < minimum)
            minimum = cloud->at(t).z;
    }
    std::cout << "Done!" << std::endl;

    return minimum;
}

bool BackgroundSegmentation::filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float threshold)
{
    PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>);
    for (auto& point : *cloud) {
        if (point.z < threshold)
            filteredCloud->push_back(point);
    }

    cloud = filteredCloud;

    return true;
}

void BackgroundSegmentation::filterBackground(Face& face)
{
    std::cout << "Looking for threshold..." << std::endl;
    float threshold = findThreshold(face.cloud);
    std::cout << "Done!" << std::endl;

    std::cout << "Removing background..." << std::endl;
    filter(face.cloud, threshold);
    std::cout << "Done!" << std::endl;
}

void BackgroundSegmentation::filterBackground(std::vector<Face>& faces)
{

    for (auto& face : faces) {
        filterBackground(face);
    }
}

/*
float BackgroundSegmentation::getTreshold() const
{
    return threshold;
}

void BackgroundSegmentation::setTreshold(float value)
{
    threshold = value;
}

cv::Mat BackgroundSegmentation::getImageRGB() const
{
    return imageRGB;
}

void BackgroundSegmentation::setImageRGB(cv::Mat& value)
{
    imageRGB = value;
}
*/

//void BackgroundSegmentation::setImageDepth(const pcl::PointCloud<pcl::PointXYZ>::Ptr& value)
//{
//    imageDepth = value;
//    num_points_ = imageDepth->size();
//}
//
//pcl::PointCloud<pcl::PointXYZ>::Ptr BackgroundSegmentation::getImageDepth() const
//{
//    return imageDepth;
//}
