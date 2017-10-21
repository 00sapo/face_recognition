#ifndef IMAGESET_H
#define IMAGESET_H
#include <boost/any.hpp>
#include <opencv2/core.hpp>

typedef unsigned int uint;
typedef cv::Matx<float, 9, 1> Pose;
namespace face {

class Image4DSetComponent {
public:
    Image4DSetComponent();

    /**
         * @brief depthForEach applies a function to every pixel
         *        in the ROI of the depth map. Function receives two coordinates (x,y) of
         *        the pixel and a reference to the pixel
         * @param function function to be called on each pixel
         * @param ROI region of interest to which apply function
         */
    template <typename T>
    void depthForEach(const std::function<void(int, int, T&)>& function, const cv::Rect& ROI)
    {
        return boost::any_cast<T>(virtualDepthForEach(function, ROI));
    }

    /**
         * @brief imageForEach applies the function function to every point
         *        in the ROI of the image. Function receives two coordinates (x,y) of
         *        the point in the image and a reference to the point
         * @param function function to be called on each point
         * @param ROI region of interest to which apply function
         */
    template <typename T>
    void imageForEach(const std::function<void(int, int, T&)>& function, const cv::Rect& ROI)
    {
        return boost::any_cast<T>(virtualImageForEach(function, ROI));
    }

    virtual cv::Mat getImage() const = 0;

    virtual cv::Mat getDepthMap() const = 0;

    /**
         * @brief getHeight gives the height (in pixels) of image and cloud
         *        (which have the same dimensions)
         * @return height of the 4D image
         */
    virtual size_t getHeight() const = 0;

    /**
         * @brief getWidth gives the width (in pixels) of image and depthmap
         *        (which have the same dimensions)
         * @return width of the 4D image
         */
    virtual size_t getWidth() const = 0;

    /**
         * @brief getArea gives the area (in pixel) of image and depthMap
         *        (which have the same dimensions)
         * @return area of the face
         */
    virtual size_t getArea() const = 0;

    /**
         * @brief get3DImage organizes the depthmap in a Mat object
         *        with 3 channels (X, Y, Z) using intrinsicMatrix.
         * @return 3D representation of the face
         */
    virtual cv::Mat get3DImage() const;

    /**
         * @brief This function crops both the image and the cloud removing
         *        every point outside the cropping region and adjusting
         *        the intrinsicMatrix accordingly to take into account the resolution change
         * @param cropRegion region of interest
         */
    virtual void crop(const cv::Rect& cropRegion);

    virtual Pose getRotationMatrix() const;
    virtual cv::Vec3f getEulerAngles() const;
    virtual cv::Vec3f getPosition() const;

    virtual void setEulerAngles(const cv::Vec3f& value);

    virtual void setPosition(const cv::Vec3f& value);

    /**
         * @brief getCloudImageRatio
         * @return downscaling ratio applied to image by the constructor
         */
    virtual float getDepthImageRatio() const;

    virtual cv::Mat getIntrinsicMatrix() const;

    virtual std::string getName();

    virtual void setName(std::string name);

protected:
    virtual boost::any virtualDepthForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI) = 0;
    virtual boost::any virtualImageForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI) = 0;

    /**
         * @brief resizeImage dowscales the image to match depth map dimensions
         */
    virtual void resizeImage();
};
}
#endif // IMAGESET_H
