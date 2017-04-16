#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H

//#include <opencv2/core/cvstd.hpp>

#include "face.h"

#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/ml/kmeans.h>
#include <pcl/point_types.h>

/**
 * @brief The BackgroundSegmentation class performs the removing of background from a RGB-D image.
 */
class BackgroundSegmentation : public pcl::Kmeans {
public:
    BackgroundSegmentation(const Face& face);

    /**
     * @brief findTreshold: Automatically finds and sets the threshold to be used for filtering
     * @return A float containing the threshold value
     */
    //float findThreshold();

    void findThreshold();

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
    bool filter(unsigned int clusterId);

    void filterBackground();

    void filterBackground(std::vector<Face>& faces);

    Face getFace() const;
    void setFace(const Face& value);

private:
    Face face;
};

#endif // BACKGROUNDSEGMENTATION_H
