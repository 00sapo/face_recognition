#ifndef POSECLUSTERIZER_H
#define POSECLUSTERIZER_H

#include <filter.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace face {

using Pose = cv::Matx<float, 9, 1>;

class PoseClusterizer : public Filter {
public:
    PoseClusterizer(int numCenters = 3);

    /**
     * @brief clusterizePoses: computes the centers of the clusters using an L2
     *        metric over the poses of input faces
     * @return true if clusterization succeded
     */
    bool clusterizePoses();

    /**
     * @brief assignFacesToClusters assigns every face to the nearest cluster center (using L2 metric)
     * @return pointer to Image4DComponent consisting of Image4DVectorComposite with cluster structure in it
     */
    void assignFacesToClusters();

    bool filter();
    Image4DComponent* getImage4DComponent() const;
    void setImage4DComponent(Image4DComponent* value);

    int getNumCenters() const;
    void setNumCenters(int value);

    std::vector<Pose> getCenters() const;
    void setCenters(const std::vector<Pose>& value);

    int getNearestCenterId(const Pose& pose);

private:
    Image4DComponent* imageSet;
    int numCenters;
    std::vector<Pose> centers;
};
}
#endif // POSECLUSTERIZER_H
