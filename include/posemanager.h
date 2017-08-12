#ifndef POSEMANAGER_H
#define POSEMANAGER_H

#include <iostream>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"

class Image4D;     // forward declaration
typedef cv::Matx<float, 9, 1> Pose;

using std::string;
using std::vector;

class PoseManager {
public:
    //PoseManager();
    explicit PoseManager(const std::string& poseEstimatorPath = POSE_ESTIMATOR_PATH);

    /**
     * @brief cropFace: crops face region taking into account face orientation
     * @param face: image containing face to crop
     * @return false if no face was detected
     */

    bool cropFaces(std::vector<Image4D>& faces, std::vector<cv::Rect> &approxFacesRegions);

    bool cropFace(Image4D& face, cv::Rect &approxFaceRegion);

    /**
     * @brief estimateFacePose
     * @param face
     * @return True if pose estimation was successful and rotation matrix was added to posesData, false otherwise
     */
    bool estimateFacePose(Image4D &face, cv::Vec3f &position, cv::Vec3f &eulerAngles);

    /**
     * @brief eulerAnglesToRotationMatrix
     * @param theta angles in radiant
     * @return Matrix 9x1 containing rotation matrix in row-major order
     */
    Pose eulerAnglesToRotationMatrix(cv::Vec3f theta);

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

    void removeOutlierBlobs(Image4D &face, const cv::Vec3f &position) const;
};

#endif // POSEMANAGER_H
