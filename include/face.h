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

    cv::Mat image; // Color or grayscale representation of the face
    cv::Mat depthMap; // Depth representation of the face

    Face();

    /**
     * @brief Face stores the two representations of the same face and shrinks
     *        the image to fit depth map dimensions. Image and depth map must have
     *        the same aspect ratio and image dimensions must be >= of depth map dimensions
     * @param image
     * @param depthMap
     */
    Face(cv::Mat image, cv::Mat depthMap);

    /**
     * @brief get3DImage organizes the cloud in a Mat object with 3 channels (X, Y, Z)
     * @return 3D representation of the face
     */
    cv::Mat get3DImage(const cv::Mat &intrinsicCameraMatrix) const ;//const;

    /**
     * @brief getWidth gives the width (in pixels) of image and cloud
     *        which have the same dimensions
     * @return width of the face
     */
    size_t getWidth()  const;

    /**
     * @brief getHeight gives the height (in pixels) of image and cloud
     *        which have the same dimensions
     * @return height of the face
     */
    size_t getHeight() const;

    /**
     * @brief getArea gives the area (in pixel) of image and depthMap
     *        which have the same dimensions
     * @return area of the face
     */
    size_t getArea() const;

    /**
     * @brief getCloudImageRatio
     * @return downscaling ratio applied to image by the constructor
     */
    float getDepthImageRatio() const;

    /**
     * @brief This function crops both the image and the cloud removing
     *        every point outside the cropping region
     * @param cropRegion region of interest
     */
    void crop(const cv::Rect &cropRegion);

    /**
     * @brief depthForEach applies a function to every pixel
     *        in the ROI of the depth map. Function receives two coordinates (x,y) of
     *        the pixel and a reference to the pixel
     * @param function function to be called on each pixel
     * @param ROI region of interest to which apply function
     */
    void depthForEach(std::function<void(int, int, float&)> function, const cv::Rect& ROI);

    void depthForEach(std::function<void(int, int, const float&)> function, const cv::Rect& ROI) const;

    /**
     * @brief imageForEach applies the function function to every point
     *        in the ROI of the image. Function receives two coordinates (x,y) of
     *        the point in the image and a reference to the point
     * @param function function to be called on each point
     * @param ROI region of interest to which apply function
     */
    void imageForEach(std::function<void(int, int, float&)> function, const cv::Rect& ROI);

    void imageForEach(std::function<void(int, int, const float&)> function, const cv::Rect& ROI) const ;

private:
    uint WIDTH;             // width of the face
    uint HEIGHT;            // height of the face
    float DEPTH_IMG_RATIO;  // downscaling ratio applied to image by the constructor

    /**
     * @brief resizeImage dowscales the image to match depth map dimensions
     */
    void resizeImage();
};

#endif // FACE_H
