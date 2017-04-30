#ifndef FACE_H
#define FACE_H

#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <functional>

typedef unsigned int uint;


/**
 * @brief The Face class represents a face with both color and 3D information
 */
class Face
{
public:

    cv::Mat image;                             // Color or grayscale representation of the face
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud; // Point cloud representation of the face

    Face();

    /**
     * @brief Face stores the two representations of the same face and shrinks
     *        the image to fit cloud dimensions. Image and cloud must have
     *        the same aspect ratio and image dimensions must be >= of cloud dimensions
     * @param image
     * @param cloud
     */
    Face(cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    /**
     * @brief get3DImage organizes the cloud in a Mat object with 3 channels (X, Y, Z)
     * @return 3D representation of the face
     */
    cv::Mat get3DImage() ;//const;

    /**
     * @brief getWidth gives the width (in pixels) of image and cloud
     *        which have the same dimensions
     * @return width of the face
     */
    uint  getWidth()  const;

    /**
     * @brief getHeight gives the height (in pixels) of image and cloud
     *        which have the same dimensions
     * @return height of the face
     */
    uint  getHeight() const;

    /**
     * @brief getCloudImageRatio
     * @return downscaling ratio applied to image by the constructor
     */
    float getCloudImageRatio() const;


    /**
     * @brief cloudForEach applies the function function to every point
     *        in the cloud. Function receives two coordinates (x,y) of
     *        the point in the organized cloud and a reference to the point
     * @param function function to be called on each point
     */
    void cloudForEach(std::function<void(uint, uint, pcl::PointXYZ &)> function);


    /**
     * @brief imageForEach applies the function function to every point
     *        in the image. Function receives two coordinates (x,y) of
     *        the point in the image and a reference to the point
     * @param function function to be called on each point
     */
    void imageForEach(std::function<void(uint, uint, float &)> function);

private:
    uint WIDTH;             // width of the face
    uint HEIGHT;            // height of the face
    float CLOUD_IMG_RATIO;  // downscaling ratio applied to image by the constructor

    /**
     * @brief resizeImage dowscales the image to match cloud dimensions
     */
    void resizeImage();
};

#endif // FACE_H
