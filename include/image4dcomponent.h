#ifndef IMAGESET_H
#define IMAGESET_H
#include <boost/any.hpp>
#include <opencv2/core.hpp>

typedef unsigned int uint;
typedef cv::Matx<float, 9, 1> Pose;
namespace face {

class Image4DComponent {
public:
    Image4DComponent(){};

    /**
         * @brief depthForEach applies a function to every pixel
         *        in the ROI of the depth map. Function receives two coordinates (x,y) of
         *        the pixel and a reference to the pixel
         * @param function function to be called on each pixel
         * @param ROI region of interest to which apply function
         */
    virtual void depthForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI) = 0;

    /**
         * @brief imageForEach applies the function function to every point
         *        in the ROI of the image. Function receives two coordinates (x,y) of
         *        the point in the image and a reference to the point
         * @param function function to be called on each point
         * @param ROI region of interest to which apply function
         */

    virtual void imageForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI) = 0;

    virtual cv::Mat getImage() const = 0;

    virtual cv::Mat getDepthMap() const = 0;

    virtual void setDepthMap(const cv::Mat& value) = 0;

    virtual void setImage(const cv::Mat& value) = 0;
    /**
         * @brief getHeight gives the height (in pixels) of the depth image
         *        (which have the same dimensions)
         * @return height of the 4D image
         */
    virtual size_t getHeight() const = 0;

    /**
         * @brief getWidth gives the width (in pixels) of the depth image
         *        (which have the same dimensions)
         * @return width of the 4D image
         */
    virtual size_t getWidth() const = 0;

    /**
         * @brief getArea gives the area (in pixel) of the depth image
         *        (which have the same dimensions)
         * @return area of the face
         */
    virtual size_t getArea() const = 0;

    /**
         * @brief get3DImage organizes the depthmap in a Mat object
         *        with 3 channels (X, Y, Z) using intrinsicMatrix.
         * @return 3D representation of the face
         */
    virtual cv::Mat get3DImage() const = 0;

    /**
         * @brief This function crops both the image and the cloud removing
         *        every point outside the cropping region and adjusting
         *        the intrinsicMatrix accordingly to take into account the resolution change
         * @param cropRegion region of interest
         */
    virtual void crop(const cv::Rect& cropRegion) = 0;

    /**
     * @brief getRotationMatrix
     * @return  a vector of Pose containing all rotation matrix of images in the set
     */
    virtual std::vector<Pose> getRotationMatrix() const = 0;
    virtual cv::Vec3f getEulerAngles() const = 0;
    virtual cv::Vec3f getPosition() const = 0;

    virtual void setEulerAngles(const cv::Vec3f& value) = 0;

    virtual void setPosition(const cv::Vec3f& value) = 0;

    /**
         * @brief getCloudImageRatio
         * @return downscaling ratio applied to image by the constructor
         */
    virtual float getDepthImageRatio() const = 0;

    virtual cv::Mat getIntrinsicMatrix() const = 0;

    virtual std::string getName() = 0;

    virtual void setName(std::string name) = 0;

    /**
         * @brief resizeImage dowscales the image to match depth map dimensions
         */
    virtual void resizeImage() = 0;

    /**
     * @brief isLeaf
     * @return true if this component is a leaf, false otherwise
     */
    virtual bool isLeaf() const = 0;

    /**
     * @brief forEachComponent execute function func on each sub component (on this component if it is a leaf)
     */
    virtual void forEachComponent(void (*func)(Image4DComponent*)) = 0;

    virtual size_t size() const = 0;

    /**
     * @brief add one item to this Image4DComponent
     * @param item the item to add
     * @return a pointer to the Image4DComponent resulting
     */
    virtual Image4DComponent* add(Image4DComponent& item) = 0;

    /**
     * @brief add add one item to the Image4DComponent located at index i of this Image4DComponent
     * @param item the item to add
     * @param i the index at which add the item
     * @return the Image4DComponent resulting
     */
    virtual Image4DComponent* add(Image4DComponent& item, uint i) = 0;

    virtual void clear() = 0;
    virtual Image4DComponent* at(uint i) = 0;

    /* method to make range based loops */
    virtual std::vector<Image4DComponent*>::iterator begin() = 0;
    virtual std::vector<Image4DComponent*>::iterator end() = 0;

    /**
     * @brief getImageCovariance
     * @return the covariance of this set of rgb images
     */
    virtual cv::Mat getImageCovariance() const = 0;
    virtual void setImageCovariance(const cv::Mat& value) = 0;

    /**
     * @brief getDepthCovariance
     * @return the covariance of this set of depth images
     */
    virtual cv::Mat getDepthCovariance() const = 0;
    virtual void setDepthCovariance(const cv::Mat& value) = 0;
};
}
#endif // IMAGESET_H
