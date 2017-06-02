#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H

#include <opencv2/objdetect.hpp>
#include "extern_libs/head_pose_estimation/CRForestEstimator.h"

#include "face.h"

/**
 * @brief The BackgroundSegmentation class performs the preprocessing
 */
class BackgroundSegmentation {
public:

    /**
     * @brief BackgroundSegmentation: constructor
     */
    BackgroundSegmentation();

    /**
     * @brief BackgroundSegmentation: constructor
     */
    BackgroundSegmentation(const std::string& faceDetectorPath, const std::string& poseEstimatorPath);

    /**
     * @brief detectForegroundFace detects the nearest face in the image
     * @param face: Face containing the image
     * @param detectedFace: ROI of the detected face
     * @return false if no face was detected, true otherwise
     */
    bool detectForegroundFace(const Face &face, cv::Rect& detectedFace);

    /**
     * @brief removeBackground splits the face cloud in two clusters, discarding
     *        furthest one
     * @param face
     */
    bool removeBackground(Face& face) const;

    /**
     * @brief removeBackground calls remove background on every Face
     * @param faces: vector of faces
     */
    void removeBackground(std::vector<Face>& faces) const;

    /**
     * @brief estimateFacePose
     * @param face
     */
    bool estimateFacePose(const Face &face);

private:

    static const std::string FACE_DETECTOR_PATH;
    static const std::string POSE_ESTIMATOR_PATH;

    bool faceDetectorAvailable  = false;
    bool poseEstimatorAvailable = false;

    cv::CascadeClassifier classifier;
    CRForestEstimator estimator;
};

#endif // BACKGROUNDSEGMENTATION_H
