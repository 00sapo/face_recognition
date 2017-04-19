#include "pointprojector.h"
#include "singletonsettings.h"
#include <opencv2/core/eigen.hpp>

PointProjector::PointProjector()
{
}

std::vector<unsigned int> PointProjector::get2DCoordinates(pcl::PointXYZ point)
{

    // Define projection matrix from 3D to 2D:
    //P matrix is in camera_info.yaml
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> P;
    Mat matP = SingletonSettings::getInstance().getP();
    cv2eigen(matP, P);

    // 3D to 2D projection:
    //Let's do P*point and rescale X,Y
    Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1);
    Eigen::Vector3f output = P * homogeneous_point;
    output[0] /= output[2];
    output[1] /= output[2];

    std::vector<unsigned int> point2D = { (unsigned int)output[1], (unsigned int)output[0] };
    return point2D;
}
