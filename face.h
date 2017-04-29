#ifndef FACE_H
#define FACE_H

#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/point_types.h>

typedef unsigned int uint;

class Face
{
public:

    cv::Mat image;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    float cloudImageRatio;

    Face();
    Face(cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    Face(cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float cloudImageRatio);

    cv::Mat get3DImage() const;
    //cv::Point2i get2DCoordinates(const pcl::PointXYZ &point) const;

    uint getWidth() const;
    uint getHeight() const;

private:
    uint WIDTH;
    uint HEIGHT;
};

#endif // FACE_H
