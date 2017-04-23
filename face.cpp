#include "face.h"
#include "pointprojector.h"

using cv::Mat;

Face::Face()
{
    image = Mat::zeros(1,1,CV_16UC3);
    cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
}

cv::Mat Face::getDepthMap()
{
    auto projector = PointProjector();
    Mat depthMap = Mat::zeros(cv::Size(image.rows, image.cols), CV_32FC1);
    for(auto& point : *cloud) {
        auto coord = projector.get2DCoordinates(point);
        if(coord[0] < 0 || coord[0] > image.rows || coord[1] < 0 || coord[1] > image.cols)
            throw std::runtime_error("out of range coordinates");
        depthMap.at<float>(coord[0], coord[1]) = point.z;
    }

    return depthMap;
}
