#include "backgroundsegmentation.h"

#include <iostream>
#include <math.h>
#include <pointprojector.h>

#include "face.h"

using pcl::PointCloud;
using pcl::PointXYZ;

BackgroundSegmentation::BackgroundSegmentation(const Face& face)
    : Kmeans(0, 1)
{
    num_clusters_ = 2;
    setFace(face);
}

void BackgroundSegmentation::findClusters()
{

    std::cout << "Adding points to KMeans..." << std::endl;

    for (unsigned int i = 0; i < face.cloud->size(); ++i) {
        float value = face.cloud->at(i).z;
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

void BackgroundSegmentation::filter(unsigned int clusterId)
{
    cv::Mat filteredImage = cv::Mat::zeros(face.image.rows, face.image.cols, CV_8UC1);
    PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>);
    for (unsigned int i; i < points_to_clusters_.size(); i++) {

        if (clusterId == points_to_clusters_[i]) {
            PointXYZ point = face.cloud->at(i);
            if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                filteredCloud->push_back(point);
                std::vector<unsigned int> xy = PointProjector::get2DCoordinates(point);
                filteredImage.at<float>(xy[0], xy[1])
                    = face.image.at<float>(xy[0], xy[1]);
            }
        }
    }

    face.cloud = filteredCloud;
    face.image = filteredImage;
    /* TODO: remove points from RGB image! */
}

void BackgroundSegmentation::filterBackground()
{
    std::cout << "Looking for threshold..." << std::endl;

    findClusters();
    std::cout << "Done!" << std::endl;

    std::cout << "Removing background..." << std::endl;
    filter(1);
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
