#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H
#include <opencv2/core/cvstd.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>

/**
 * @brief The BackgroundSegmentation class performs the removing of background from a RGB-D image.
 */
class BackgroundSegmentation {
public:
    BackgroundSegmentation();

    /**
     * @brief findTreshold: Automatically finds the threshold to be used for filtering
     * @return A double containing the threshold value
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

    std::vector<cv::Mat> getCollectionRGB() const;
    void setCollectionRGB(const std::vector<cv::Mat>& value);

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> getCollectionDepth() const;
    void setCollectionDepth(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& value);

private:
    /**
     * @brief collectionRGB is a vector containing all the images with RGB values
     */
    std::vector<cv::Mat> collectionRGB;

    /**
     * @brief collectionDepth is a vector containing all the images with Depth values
     */
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> collectionDepth;

    float treshold;
};

#endif // BACKGROUNDSEGMENTATION_H
