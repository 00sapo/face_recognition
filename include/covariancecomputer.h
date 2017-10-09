#ifndef FACE_COVARIANCECOMPUTER_H
#define FACE_COVARIANCECOMPUTER_H

#include <vector>
#include <iostream>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"

namespace face {

class Face;

using Pose = cv::Matx<float, 9, 1>;


/*
 * TODO: Covariance computer class could be static, it is simply a collection of functions
 */
class CovarianceComputer {
public:
    CovarianceComputer();

    /**
     * @brief computeCovarianceRepresentation: extracts a covariance matrix based representation
     *        of an input Face set. Input faces are clusterized in subsets based on their pose
     *        and then for each subset a pair of covariance matrixes, one for images and the other
     *        for the depth maps, representative of the set are computed
     * @param faces: input face set
     * @param subsets: number of desired clusters
     * @return a pair of covariance matrixes for each cluster
     */
    std::vector<std::pair<cv::Mat, cv::Mat>> computeCovarianceRepresentation(const std::vector<Face> &faces, int subsets) const;


    /**
     * @brief clusterizePoses: computes the centers of the clusters using an L2
     *        metric over the poses of input faces
     * @param numCenters: num of centers/clusters
     * @return a vector containing clusters centers
     */
    std::vector<Pose> clusterizePoses(const std::vector<Face>& faces, int numCenters) const;

    /**
     * @brief assignFacesToClusters assigns every face to the nearest cluster center (using L2 metric)
     * @param faces: vector of faces to cluster
     * @param centers: cluster centers
     * @return vector containing in each position a list of faces assigned to that cluster
     */
    std::vector<std::vector<const Face*>> assignFacesToClusters(const std::vector<Face> &faces,
                                                                const std::vector<Pose> &centers) const;

    /**
     * @brief setToCovariance: computes the normalized covariance matrix representation of an image set
     * @param set: face set
     * @param imageCovariance: output covariance matrix for the images in the set
     * @param depthCovariance: output covariance matric for the depth maps in the set
     */
    void setToCovariance(const std::vector<const Face*> &set,
                         cv::Mat& imageCovariance,
                         cv::Mat& depthCovariance) const;

private:
    /**
     * @brief getNearestCenterId
     * @param poseEstimation
     * @return id of the nearest center to the input pose estimation
     */
    int getNearestCenterId(const Pose& pose, const std::vector<Pose>& centers) const;

};

} // namespace face

#endif // FACE_COVARIANCECOMPUTER_H
