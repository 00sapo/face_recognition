#ifndef POSEMANAGER_H
#define POSEMANAGER_H
#include "extern_libs/head_pose_estimation/CRForestEstimator.h"
#include "face.h"
#include "iostream"
#include "singletonsettings.h"

using std::string;
using std::vector;

class PoseManager {
public:
    PoseManager();
    PoseManager(const std::string& poseEstimatorPath);

    /**
     * @brief estimateFacePose
     * @param face
     * @return True if pose estimation was successful and rotation matrix was added to posesData, false otherwise
     */
    bool estimateFacePose(const Face& face);

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

    vector<cv::Mat> centers;

    vector<cv::Mat> posesData;
};

#endif // POSEMANAGER_H
