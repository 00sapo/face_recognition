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

    //explicit PoseManager(const std::vector<Face> &faces);

    std::vector<cv::Mat> computeCovarianceRepresentation(const std::vector<Face> &faces, int subsets);

    /**
     * @brief eulerAnglesToRotationMatrix
     * @param theta angles in radiant
     * @return Matrix 9x1 containing rotation matrix in row-major order
     */
    static Pose eulerAnglesToRotationMatrix(const cv::Vec3f &theta);

    /**
     * @brief addPoseData add pose to the dataset on which perform clustering
     * @param pose
     */
    void addPoseData(const Pose &pose);

private:

    /**
     * @brief clusterizePoses
     * @param numCenters num of centers/clusters
     * @return true if clusterization succeeded, false otherwise
     */
    std::vector<Pose> clusterizePoses(const std::vector<Face> &faces, int numCenters) const;

    /**
     * @brief assignFacesToClusters assigns every face to the nearest cluster center (using L2 metric)
     * @param faces: vector of faces to cluster
     * @param centers: cluster centers
     * @return vector
     */
    std::vector<std::vector<const Face*>> assignFacesToClusters(const std::vector<Face> &faces,
                                                                const std::vector<Pose> &centers) const;

    /**
     * @brief getNearestCenterId
     * @param poseEstimation
     * @return id of the nearest center to the input pose estimation
     */
    int getNearestCenterId(const Pose &pose, const std::vector<Pose> &centers) const;

    /**
     * @brief setToCovariance: computes the covariance matrix representation of an image set
     * @param set: image set
     * @return covariance matrix
     */
    cv::Mat setToCovariance(const std::vector<const Face*> &set) const;



    //cv::Mat centers;
    //
    //std::vector<Pose> posesData;

};

}   // face

#endif // POSEMANAGER_H
