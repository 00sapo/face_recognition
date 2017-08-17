#include "face.h"

#include "image4d.h"

using cv::Mat;

namespace face {

Face::Face() : Image4D() { }

Face::Face(const Image4D &image, const cv::Vec3f &position, const cv::Vec3f &eulerAngles)
    : Image4D(image), position(position), eulerAngles(eulerAngles) { }

Pose Face::getRotationMatrix()
{
    // Calculate rotation around x axis
    float cosx = cos(eulerAngles[0]);
    float senx = sin(eulerAngles[0]);
    float cosy = cos(eulerAngles[1]);
    float seny = sin(eulerAngles[1]);
    float cosz = cos(eulerAngles[2]);
    float senz = sin(eulerAngles[2]);

    return Pose(cosy * cosz, cosx * senz + senx * seny * cosz, senx * senz - cosx * seny * cosz,
        -cosy * senz, cosx * cosz - senx * seny * senz, senx * cosz + cosx * seny * senz,
        seny, -senx * cosy, cosx * cosy);
}

cv::Vec3f Face::getEulerAngles() { return eulerAngles; }
cv::Vec3f Face::getPosition()    { return position;    }

}
