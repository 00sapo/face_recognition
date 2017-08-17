#ifndef POSEMANAGER_H
#define POSEMANAGER_H

#include <iostream>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"


namespace face {

class Image4D;     // forward declaration
class Face;
typedef cv::Matx<float, 9, 1> Pose;


class PoseManager {
public:
    explicit PoseManager(const std::string &poseEstimatorPath = POSE_ESTIMATOR_PATH);

    /**
     * @brief cropFace: crops face region taking into account face orientation
     * @param face: image containing face to crop
     * @return false if no face was detected
     */

    std::vector<face::Face> cropFaces(std::vector<face::Image4D> &faces);

    bool cropFace(face::Image4D &image4d, cv::Vec3f &position, cv::Vec3f &eulerAngles);

    /**
     * @brief estimateFacePose
     * @param face
     * @return True if pose estimation was successful and rotation matrix was added to posesData, false otherwise
     */
    bool estimateFacePose(const face::Image4D &image4d, cv::Vec3f &position, cv::Vec3f &eulerAngles);

    /**
     * @brief eulerAnglesToRotationMatrix
     * @param theta angles in radiant
     * @return Matrix 9x1 containing rotation matrix in row-major order
     */
    face::Pose eulerAnglesToRotationMatrix(cv::Vec3f theta);

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
    int getNearestCenterId(face::Pose poseEstimation);

    /**
     * @brief addPoseData add pose to the dataset on which perform clustering
     * @param pose
     */
    void addPoseData(face::Pose pose);

private:
    static const std::string POSE_ESTIMATOR_PATH;

    bool poseEstimatorAvailable = false;

    CRForestEstimator estimator;

    cv::Mat centers;

    std::vector<face::Pose> posesData;

};

}   // face

#endif // POSEMANAGER_H
