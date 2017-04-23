#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/ml/kmeans.h>
#include <pcl/point_types.h>

#include "face.h"

/**
 * @brief The BackgroundSegmentation class performs the removing of background from a RGB-D image.
 */
class BackgroundSegmentation : public pcl::Kmeans {
public:
    /**
     * @brief BackgroundSegmentation: constructor
     * @param face: the face to be used
     */
    BackgroundSegmentation(const Face& face);

    /**
     * @brief findClusters: finds clusters in the cloud of the face
     */

    void findClusters();

    /**
     * @brief filter: remove face cloud the points that are not in the cluster specified
     * @param clusterId: the id of the cluster, it can be 0 or 1
     * @return
     */
    void filter(unsigned int clusterId);

    /**
     * @brief filterBackground: finds clusters and then calls filter(1)
     */
    void filterBackground();

    /**
     * @brief filterBackground: same as filterBackground() but for every face in the vector
     * @param faces
     */
    void filterBackground(std::vector<Face>& faces);

    Face getFace() const;
    void setFace(const Face& value);

private:
    /**
     * @brief face: the face that contains the RGB-D image to be process
     */
    Face face;

    void cropFace();
};

#endif // BACKGROUNDSEGMENTATION_H
