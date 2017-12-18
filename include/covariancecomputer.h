#ifndef FACE_COVARIANCECOMPUTER_H
#define FACE_COVARIANCECOMPUTER_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace face {

using Pose = cv::Matx<float, 9, 1>;

class Face;


/**
 * @brief A collection of functions related to compute a covariance matrix representation
 *        from a Face set. computeCovarianceRepresentation() does the job, all the other
 *        functions are used by it.
 */
namespace covariance {


    void getNormalizedCovariances(const std::vector<Face>& identity, int subsets, std::vector<cv::Mat>& grayscaleCovarOut,
        std::vector<cv::Mat>& depthmapCovarOut);


    /**
     * @brief computeCovarianceRepresentation: extracts a covariance matrix based representation
     *        of an input Face set. Input faces are clusterized in subsets based on their pose
     *        and then for each subset a pair of covariance matrixes, one for images and the other
     *        for the depth maps, representative of the set are computed
     * @param faces: input face set
     * @param subsets: number of desired clusters
     * @return a pair of covariance matrixes for each cluster
     */
    std::vector<std::pair<cv::Mat, cv::Mat>> computeCovarianceRepresentation(const std::vector<Face> &faces, int subsets);


    /**
     * @brief clusterizePoses: computes the centers of the clusters using an L2
     *        metric over the poses of input faces
     * @param numCenters: num of centers/clusters
     * @return a vector containing clusters centers (sorted in increasing order)
     */
    std::vector<float> clusterizePoses(const std::vector<Face> &faces, int numCenters);

    /**
     * @brief assignFacesToClusters assigns every face to the nearest cluster center (using L2 metric)
     * @param faces: vector of faces to cluster
     * @param centers: cluster centers
     * @return vector containing in each position a list of faces assigned to that cluster
     */
    std::vector<std::vector<const Face*>> assignFacesToClusters(const std::vector<Face> &faces,
                                                                const std::vector<float> &centers);

    /**
     * @brief setToCovariance: computes the covariance matrix representation of an image set
     * @param set: face set
     * @param imageCovariance: output covariance matrix for the images in the set
     * @param depthCovariance: output covariance matric for the depth maps in the set
     */
    void setToCovariance(const std::vector<const Face*> &set,
                         cv::Mat &imageCovariance,
                         cv::Mat &depthCovariance);

} // namespace covariance

} // namespace face

#endif // COVARIANCECOMPUTER_H
