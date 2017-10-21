#ifndef FACE_IMAGE_4D_H
#define FACE_IMAGE_4D_H

#include <functional>
#include <image4dsetcomponent.h>
#include <opencv2/opencv.hpp>

namespace face {

/**
     * @brief The Image4DLeaf class is a couple (grayscale image - depth map)
     *        that represents both the visual and spatial information of a scene
     */
class Image4DLeaf : public Image4DSetComponent {
public:
    Image4DLeaf();

    /**
         * @brief Image4dLeaf stores two representations of the same image (depth and rgb) and shrinks
         *        the image to fit depth map dimensions. Image and depth map must have
         *        the same aspect ratio and image dimensions must be >= of depth map dimensions
         * @param image
         * @param depthMap
         * @param intrinsicCameraMatrix
         */
    Image4DLeaf(cv::Mat& image, cv::Mat& depthMap, const cv::Mat& intrinsicCameraMatrix);

    cv::Mat get3DImage() const;

    size_t getWidth() const;

    size_t getHeight() const;

    size_t getArea() const;

    float getDepthImageRatio() const;

    cv::Mat getIntrinsicMatrix() const;

    std::string getName()
    {
        return name;
    }

    void setName(std::string name)
    {
        this->name = name;
    }

    void crop(const cv::Rect& cropRegion);

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

    cv::Mat getImage() const;

    cv::Mat getDepthMap() const;

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

    void resizeImage();
};

} // face

#endif // IMAGE_4D_H_
