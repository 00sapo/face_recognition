#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H

//#include <opencv2/core/cvstd.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/ml/kmeans.h>
#include <pcl/point_types.h>

class Face;

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
    //float findThreshold();

    float findThreshold(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    /**
     * @brief filter: Remove the background in each image of the collection using the threshold
     * @return return True if everything worked, false otherwise.
     */
    //bool filter();

    /**
     * @brief filter: set the threshold and calls filter()
     * @param threshold: the value to use as threshold
     * @return same as filter
     */
    bool filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float threshold);

    void filterBackground(Face& face);

    void filterBackground(std::vector<Face>& faces);
};

#endif // BACKGROUNDSEGMENTATION_H
