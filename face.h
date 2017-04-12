#ifndef FACE_H
#define FACE_H

#include <pcl/common/common.h>
#include <pcl/point_types.h>

class Face
{
public:

    Mat imageRGB;
    pcl::PointCloud<pcl::PointXYZ> imageD;

    Face();
};

#endif // FACE_H
