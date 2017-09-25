#ifndef FACE_PREPROCESSOR_H
#define FACE_PREPROCESSOR_H

#include <atomic>
#include <mutex>
#include <opencv2/objdetect.hpp>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"

namespace face {
struct threshold {
    float freq;
    uint16_t minTh;
    uint16_t maxTh;
};

class Image4D; // forward declarations
class Face;

class Preprocessor {
public:
    Preprocessor(const std::string& faceDetectorPath = FACE_DETECTOR_PATH, const std::string& poseEstimatorPath = POSE_ESTIMATOR_PATH);

    /**
     * @brief preprocess cleans up and prepares the 4D images to be processed
     * @param images
     * @return a vector of cropped faces
     */
    std::vector<Face> preprocess(std::vector<Image4D>& images);

    /**
     * @brief segment is the first preprocessing step. Removes the background
     *        from the image
     * @param faces
     * @return true if all images background have been removed successfully
     */

    void findDepthThresholds(Image4D& image4d, std::vector<threshold>& interestingThs, int k);

    /**
     * @brief cropFaces is the second preprocessing step. Precisely
     *        crops the region around the detected face
     * @param faces
     * @return a vector containing the cropped faces
     */
    std::vector<face::Face> cropFaces(std::vector<face::Image4D>& images);

    int findLeastFreqTh(std::vector<threshold> interestingThs);

private:
    static const std::string FACE_DETECTOR_PATH;
    static const std::string POSE_ESTIMATOR_PATH;

    bool faceDetectorAvailable = false;
    bool poseEstimatorAvailable = false;

    cv::CascadeClassifier classifier;
    CRForestEstimator estimator;

    /**
     * @brief detectForegroundFace detects the nearest face in the image
     * @param face: Face containing the image
     * @param detectedFace: ROI of the detected face
     * @return false if no face was detected, true otherwise
     */
    bool detectForegroundFace(const Image4D& face, cv::Rect& boundingBox);

    /**
     * @brief removeBackground splits the face cloud in two clusters, discarding
     *        furthest one
     * @param face
     */
    void removeBackgroundDynamic(Image4D& face, const cv::Rect& boundingBox) const;

    void removeBackgroundFixed(Image4D& face, uint16_t threshold) const;

    /**
     * @brief cropFace: crops face region taking into account face orientation
     * @param face: image containing face to crop
     * @return false if no face was detected
     */
    bool cropFace(face::Image4D& image4d, cv::Vec3f& position, cv::Vec3f& eulerAngles) const;

    /**
     * @brief estimateFacePose
     * @param face
     * @return True if pose estimation was successful and rotation matrix was added to posesData, false otherwise
     */
    bool estimateFacePose(const face::Image4D& image4d, cv::Vec3f& position, cv::Vec3f& eulerAngles) const;
    bool filterAndCrop(Image4D& face, uint16_t minTh, uint16_t maxTh, cv::Vec3f& position, cv::Vec3f& eulerAngles);
};

} // namespace face

#endif // FACE_PREPROCESSOR_H
