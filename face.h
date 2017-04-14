#ifndef FACE_H
#define FACE_H

#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/point_types.h>


class Face
{
public:

    cv::Mat image;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

    Face();
};

#endif // FACE_H
