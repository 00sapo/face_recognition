#include "backgroundsegmentation.h"

#include <iostream>
#include <math.h>
#include <pointprojector.h>

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
    unsigned int clusterId = 0;
    for (int i = 0; i < num_clusters_; i++) {

        if (centroids_[i][0] < min[0]) {
            clusterId = i;
            min = centroids_[i];
        }
    }

    cv::Mat filteredImage = cv::Mat::zeros(face.image.rows, face.image.cols, CV_8U);
    PointCloud<PointXYZ>::Ptr filteredCloud(new PointCloud<PointXYZ>);
    for (unsigned int i; i < points_to_clusters_.size(); i++) {

        if (clusterId == points_to_clusters_[i]) {
            PointXYZ point = face.cloud->at(i);
            if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
                filteredCloud->push_back(point);
                std::vector<int> xy = { i / face.cloud->width, i % face.cloud->width / 4 };
                filteredImage.at<int>(xy[0], xy[1])
                    = face.image.at<int>(xy[0], xy[1]);
            }
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
