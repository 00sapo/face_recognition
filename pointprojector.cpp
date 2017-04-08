#include "pointprojector.h"
#include "singletonsettings.h"

PointProjector::PointProjector()
{
}

static std::vector<float> PointProjector::get2DCoordinates(pcl::PointXYZRGB point)
{
    // Define projection matrix from 3D to 2D:
    //P matrix is in camera_info.yaml
    Mat P = SingletonSettings.getInstance()->getP();
    //    Eigen::Matrix<float, 3, 4> P(cvP.data());

    // 3D to 2D projection:
    //Let's do P*point and rescale X,Y
    Vec4f homogeneous_point = (point.x, point.y, point.z, 1);
    Mat output = P * Mat(homogeneous_point);
    output[0] /= output[2];
    output[1] /= output[2];

    std::vector<float> point2D = { output[0], output[1] };
    return point2D;
}
