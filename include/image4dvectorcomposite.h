#ifndef CLUSTEROFIMAGES_H
#define CLUSTEROFIMAGES_H
#include <image4dcomponent.h>
#include <image4dleaf.h>
#include <vector>

using std::vector;

namespace face {
class Image4DVectorComposite : public Image4DComponent {

public:
    Image4DVectorComposite();
    /**
     * @brief getImage returns the first rgb image of this vector
     * @return
     */
    cv::Mat getImage() const;

    /**
     * @brief getDepthMap returns the first depth map of this vector
     * @return
     */
    cv::Mat getDepthMap() const;

    /**
     * @brief getHeight
     * @return the height of the first image of this vector
     */
    size_t getHeight() const;

    /**
     * @brief getWidth
     * @return the width of the first image of this vector
     */
    size_t getWidth() const;

    /**
     * @brief getArea
     * @return the area of the first image of this vector
     */
    size_t getArea() const;

    /**
     * @brief get3DImage
     * @return the 3d image of the first image of this vector
     */
    cv::Mat get3DImage() const;

    /**
     * @brief crop crop all images in the vector
     * @param cropRegion
     */
    void crop(const cv::Rect& cropRegion);

    /**
     * @brief getRotationMatrix
     * @return the rotation matrix of the first image of this vector
     */
    vector<Pose> getRotationMatrix() const;

    /**
     * @brief getEulerAngles
     * @return the euler angles of the first image of this vector
     */
    cv::Vec3f getEulerAngles() const;

    /**
     * @brief getPosition
     * @return position of the first image of this vector
     */
    cv::Vec3f getPosition() const;

    /**
     * @brief setEulerAngles set euler angles for all the images in the vector
     * @param value
     */
    void setEulerAngles(const cv::Vec3f& value);

    /**
     * @brief setPosition set the position for all the images in the vector
     * @param value
     */
    void setPosition(const cv::Vec3f& value);

    /**
     * @brief getDepthImageRatio
     * @return the ratio of depth image of the first image of this vector
     */
    float getDepthImageRatio() const;

    /**
     * @brief getIntrinsicMatrix
     * @return the intrinsic matrix of the first image of this vector
     */
    cv::Mat getIntrinsicMatrix() const;

    /**
     * @brief getName
     * @return the name of the first image of this vector
     */
    std::string getName();

    /**
     * @brief setName
     * @param name set the name of the first image of this vector
     */
    void setName(std::string name);

    /**
     * @brief resizeImage resize all images of the vector
     */
    void resizeImage();

    /**
     * @brief isLeaf
     * @return false
     */
    bool isLeaf() const;

    void forEachComponent(void (*func)(Image4DComponent*));

    vector<Image4DComponent*> getVec() const;
    void setVec(const vector<Image4DComponent*>& value);

    size_t size() const;

    Image4DComponent* add(Image4DComponent& item);
    Image4DComponent* add(Image4DComponent& item, uint i);

    vector<Image4DComponent*>::iterator begin();
    vector<Image4DComponent*>::iterator end();

    void clear();
    Image4DComponent* at(uint i);

    cv::Mat getImageCovariance() const;
    void setImageCovariance(const cv::Mat& value);

    cv::Mat getDepthCovariance() const;
    void setDepthCovariance(const cv::Mat& value);

    /**
     * @brief virtualDepthForEach depthForEach on all images of the vector
     * @param function
     * @param ROI
     * @return
     */
    void depthForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI);

    /**
     * @brief virtualImageForEach
     * @param function
     * @param ROI
     * @return imageForEach on all images of the vector
     */
    void imageForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI);

protected:
    vector<Image4DComponent*> vec;
    cv::Mat depthCovariance;
    cv::Mat imageCovariance;
};
}
#endif // CLUSTEROFIMAGES_H
