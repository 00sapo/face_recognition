#ifndef POSEMANAGER_H
#define POSEMANAGER_H

#include <iostream>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"

class Face;     // forward declaration
typedef cv::Matx<float, 9, 1> Pose;

using std::string;
using std::vector;

class PoseManager {
public:
    //PoseManager();
    explicit PoseManager(const std::string& poseEstimatorPath = POSE_ESTIMATOR_PATH);

    /**
     * @brief estimateFacePose
     * @param face
     * @return True if pose estimation was successful and rotation matrix was added to posesData, false otherwise
     */
    bool estimateFacePose(const Face& face, const cv::Mat &calibration);

    /**
     * @brief eulerAnglesToRotationMatrix
     * @param theta angles in radiant
     * @return Matrix 9x1 containing rotation matrix in row-major order
     */
    Pose eulerAnglesToRotationMatrix(Vec3f theta);

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
    int getNearestCenterId(Pose poseEstimation);

    /**
     * @brief addPoseData add pose to the dataset on which perform clustering
     * @param pose
     */
    void addPoseData(Pose pose);

private:
    static const std::string POSE_ESTIMATOR_PATH;

    bool poseEstimatorAvailable = false;

    CRForestEstimator estimator;

    cv::Mat centers;

    vector<Pose> posesData;
};

#endif // POSEMANAGER_H
