#ifndef FACE_H_
#define FACE_H_

#include "image4d.h"

namespace face {


class Image4D;  // forward declaration
typedef cv::Matx<float, 9, 1> Pose;


/**
 * @brief The Face class extends Image4D adding informations about
 *        position and rotation of the face
 */
class Face : public Image4D
{

public:
    Face();
    Face(const Image4D &image, const cv::Vec3f &position, const cv::Vec3f &eulerAngles);

    Pose getRotationMatrix();
    cv::Vec3f getEulerAngles();
    cv::Vec3f getPosition();

private:

    cv::Vec3f eulerAngles;
    cv::Vec3f position;
};

}   // namespace face

#endif // FACE_H_
