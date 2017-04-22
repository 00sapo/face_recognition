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
    //auto projector = PointProjector();
    Mat depthMap = Mat::zeros(image.rows, image.cols, CV_8UC1);
    //for(auto& point : *cloud) {
    //    auto coord = projector.get2DCoordinates(point);
    //    depthMap.at<unsigned int>(coord[0], coord[1]) = point.z;
    //}

    return depthMap;
}
