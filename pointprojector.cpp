#include "pointprojector.h"
#include "singletonsettings.h"
#include <opencv2/core/eigen.hpp>

PointProjector::PointProjector()
{
}

std::vector<float> PointProjector::get2DCoordinates(pcl::PointXYZ point)
{

    // Define projection matrix from 3D to 2D:
    //P matrix is in camera_info.yaml
    Eigen::Matrix<float, 3, 4> P;
    //    Mat matP = SingletonSettings::getInstance().getP();
    //    std::string pmatrix;
    //    for (int i = 0; i < matP.rows; ++i) {
    //        for (int j = 0; j < matP.cols; ++j) {
    //            pmatrix << matP.at<float>(i, j) << ", ";
    //        }
    //    }
    //    P << pmatrix;

    // 3D to 2D projection:
    //Let's do P*point and rescale X,Y
    Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1);
    Eigen::Vector3f output = P * homogeneous_point;
    output[0] /= output[2];
    output[1] /= output[2];

    //    // 3D to 2D projection:
    //    //Let's do P*point and rescale X,Y
    //    Vec4f homogeneous_point(point.x, point.y, point.z, 1);
    //    Mat output = P * Mat(homogeneous_point);
    //    output.at<float>(0, 0) /= output.at<float>(2, 0);
    //    output.at<float>(1, 0) /= output.at<float>(2, 0);

    std::vector<float> point2D = { output[0], output[1] };
    return point2D;
}
