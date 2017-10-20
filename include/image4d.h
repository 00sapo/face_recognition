#ifndef FACE_IMAGE_4D_H
#define FACE_IMAGE_4D_H

#include <functional>
#include <imageset.h>
#include <opencv2/opencv.hpp>

typedef unsigned int uint;
typedef cv::Matx<float, 9, 1> Pose;

namespace face {

/**
     * @brief The Image4D class is a couple (grayscale image - depth map)
     *        that represents both the visual and spatial information of a scene
     */
class Image4D : public Image4DSet {
public:
    Image4D();

    /**
         * @brief Face stores the two representations of the same face and shrinks
         *        the image to fit depth map dimensions. Image and depth map must have
         *        the same aspect ratio and image dimensions must be >= of depth map dimensions
         * @param image
         * @param depthMap
         * @param intrinsicCameraMatrix
         */
    Image4D(cv::Mat& image, cv::Mat& depthMap, const cv::Mat& intrinsicCameraMatrix);

    /**
     * @brief Image4D create an Image4D using one already existing and new position and eulerAngles
     * @param image
     * @param position
     * @param eulerAngles
     */
    Image4D(Image4D& image, cv::Vec3f& position, cv::Vec3f& eulerAngles);

    /**
         * @brief get3DImage organizes the depthmap in a Mat object
         *        with 3 channels (X, Y, Z) using intrinsicMatrix.
         * @return 3D representation of the face
         */
    cv::Mat get3DImage() const;

    /**
         * @brief getWidth gives the width (in pixels) of image and depthmap
         *        (which have the same dimensions)
         * @return width of the 4D image
         */
    size_t getWidth() const;

    /**
         * @brief getHeight gives the height (in pixels) of image and cloud
         *        (which have the same dimensions)
         * @return height of the 4D image
         */
    size_t getHeight() const;

    /**
         * @brief getArea gives the area (in pixel) of image and depthMap
         *        (which have the same dimensions)
         * @return area of the face
         */
    size_t getArea() const;

    /**
         * @brief getCloudImageRatio
         * @return downscaling ratio applied to image by the constructor
         */
    float getDepthImageRatio() const;

    cv::Mat getIntrinsicMatrix() const;

    std::string getName()
    {
        return name;
    }
    /**
         * @brief This function crops both the image and the cloud removing
         *        every point outside the cropping region and adjusting
         *        the intrinsicMatrix accordingly to take into account the resolution change
         * @param cropRegion region of interest
         */
    void crop(const cv::Rect& cropRegion);

    /**
         * @brief depthForEach applies a function to every pixel
         *        in the ROI of the depth map. Function receives two coordinates (x,y) of
         *        the pixel and a reference to the pixel
         * @param function function to be called on each pixel
         * @param ROI region of interest to which apply function
         */
    boost::any virtualDepthForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI)
    {

        const uint MAX_X = ROI.x + ROI.width;
        const uint MAX_Y = ROI.y + ROI.height;

        for (uint y = ROI.y; y < MAX_Y; ++y) {
            for (uint x = ROI.x; x < MAX_X; ++x) {
                function(x, y, depthMap.at<boost::any>(y, x));
            }
        }
    }

    /**
         * @brief imageForEach applies the function function to every point
         *        in the ROI of the image. Function receives two coordinates (x,y) of
         *        the point in the image and a reference to the point
         * @param function function to be called on each point
         * @param ROI region of interest to which apply function
         */
    boost::any virtualImageForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI)
    {
        const uint MAX_X = ROI.x + ROI.width;
        const uint MAX_Y = ROI.y + ROI.height;

        for (uint y = ROI.y; y < MAX_Y; ++y) {
            for (uint x = ROI.x; x < MAX_X; ++x) {
                function(x, y, image.at<boost::any>(y, x));
            }
        }
    }

    Pose getRotationMatrix() const;
    cv::Vec3f getEulerAngles() const;
    cv::Vec3f getPosition() const;

    void setEulerAngles(const cv::Vec3f& value);

    void setPosition(const cv::Vec3f& value);

    const cv::Mat* getImage() const;

    const cv::Mat* getDepthMap() const;

private:
    cv::Vec3f eulerAngles;
    cv::Vec3f position;
    cv::Mat image; // Color or grayscale representation of the face
    cv::Mat depthMap; // Depth representation of the face
    std::string name;

protected:
    uint WIDTH; // width of the face
    uint HEIGHT; // height of the face
    float DEPTH_IMG_RATIO; // downscaling ratio applied to image by the constructor

    cv::Mat intrinsicMatrix;

    /**
         * @brief resizeImage dowscales the image to match depth map dimensions
         */
    void resizeImage();
};

} // face

#endif // IMAGE_4D_H_
