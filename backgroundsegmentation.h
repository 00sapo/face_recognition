#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H
#include <opencv2/core/cvstd.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/ml/kmeans.h>
#include <pcl/point_types.h>

/**
 * @brief The BackgroundSegmentation class performs the removing of background from a RGB-D image.
 */
class BackgroundSegmentation : public pcl::Kmeans {
public:
    BackgroundSegmentation();

    /**
     * @brief findTreshold: Automatically finds and sets the threshold to be used for filtering
     * @return A float containing the threshold value
     */
    float findTreshold();

    /**
     * @brief filter: Remove the background in each image of the collection using the threshold
     * @return return True if everything worked, false otherwise.
     */
    bool filter();

    /**
     * @brief filter: set the threshold and calls filter()
     * @param threshold: the value to use as threshold
     * @return same as filter
     */
    bool filter(double threshold);

    float getTreshold() const;
    void setTreshold(float value);

    cv::Mat getImageRGB() const;
    void setImageRGB(cv::Mat& value);

    pcl::PointCloud<pcl::PointXYZ>::Ptr getImageDepth() const;
    void setImageDepth(const pcl::PointCloud<pcl::PointXYZ>::Ptr& value);

private:
    /**
     * @brief collectionRGB is a vector containing all the images with RGB values
     */
    cv::Mat imageRGB;

    /**
     * @brief collectionDepth is a vector containing all the images with Depth values
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr imageDepth;

    float threshold;
};

#endif // BACKGROUNDSEGMENTATION_H
