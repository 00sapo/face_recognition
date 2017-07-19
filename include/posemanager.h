#ifndef POSEMANAGER_H
#define POSEMANAGER_H

#include <iostream>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"

class Face;     // forward declaration

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
     * @return Matrix 1x9 containing rotation matrix in row-major order
     */
    cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f& theta);

    bool clusterizePoses(uint numCenters);

    uint getNearestCenterId(cv::Mat estimation);

    void addPoseData(cv::Mat pose);

private:
    static const std::string POSE_ESTIMATOR_PATH;

    bool poseEstimatorAvailable = false;

    CRForestEstimator estimator;

    std::vector<cv::Mat> centers;

    std::vector<cv::Mat> posesData;
};

#endif // POSEMANAGER_H
