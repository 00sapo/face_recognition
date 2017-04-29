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
    float cloudImageRatio;

    Face();
    Face(cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    Face(cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float cloudImageRatio);

    cv::Mat get3DImage();
    cv::Point2i get2DCoordinates(const pcl::PointXYZ &point) const;
};

#endif // FACE_H
