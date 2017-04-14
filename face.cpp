#include "face.h"

Face::Face()
{
    image = cv::Mat::zeros(1,1,CV_16UC3);
    cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
}

