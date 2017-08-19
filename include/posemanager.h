#ifndef POSEMANAGER_H
#define POSEMANAGER_H

#include <iostream>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"


namespace face {

class Face;
typedef cv::Matx<float, 9, 1> Pose;


class PoseManager {
public:

    PoseManager();

    explicit PoseManager(const std::vector<Face> &faces);

    /**
     * @brief eulerAnglesToRotationMatrix
     * @param theta angles in radiant
     * @return Matrix 9x1 containing rotation matrix in row-major order
     */
    static Pose eulerAnglesToRotationMatrix(const cv::Vec3f &theta);

    /**
     * @brief clusterizePoses
     * @param numCenters num of centers/clusters
     * @return true if clusterization succeeded, false otherwise
     */
    bool clusterizePoses(int numCenters);

    /**
     * @brief getNearestCenterId
     * @param poseEstimation
     * @return id of the nearest center to the input pose estimation
     */
    int getNearestCenterId(Pose &poseEstimation);

    /**
     * @brief addPoseData add pose to the dataset on which perform clustering
     * @param pose
     */
    void addPoseData(const Pose &pose);

private:

    cv::Mat centers;

    std::vector<Pose> posesData;

};

}   // face

#endif // POSEMANAGER_H
