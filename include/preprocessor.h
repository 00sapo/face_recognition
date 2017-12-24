#ifndef FACE_PREPROCESSOR_H
#define FACE_PREPROCESSOR_H

#include <atomic>
#include <mutex>
#include <opencv2/objdetect.hpp>

#include "extern_libs/head_pose_estimation/CRForestEstimator.h"

namespace face {

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
    std::vector<Face> preprocess(std::vector<Image4D> images);

    /**
     * @brief segment is the first preprocessing step. Removes the background
     *        from the image
     * @param faces
     * @return true if all images background have been removed successfully
     */
    void segment(std::vector<Image4D>& images);

    void segment(Image4D& face);

    /**
     * @brief cropFaces is the second preprocessing step. Precisely
     *        crops the region around the detected face
     * @param faces
     * @return a vector containing the cropped faces
     */
    std::vector<face::Face> cropFaces(std::vector<face::Image4D>& images);

    /**
     * @brief cropFace: crops face region taking into account face orientation
     * @param face: image containing face to crop
     * @return false if no face was detected
     */
    bool cropFace(const Image4D& image4d, Face& croppedFace);

    /**
     * @brief estimateFacePose
     * @param face
     * @return True if pose estimation was successful and rotation matrix was added to posesData, false otherwise
     */
    bool estimateFacePose(const face::Image4D& image4d, cv::Vec3f& position, cv::Vec3f& eulerAngles);

private:
    static const std::string FACE_DETECTOR_PATH;
    static const std::string POSE_ESTIMATOR_PATH;

    static constexpr float FIXED_THRESHOLD = 1600;

    bool faceDetectorAvailable = false;
    bool poseEstimatorAvailable = false;

    cv::CascadeClassifier classifier;
    CRForestEstimator estimator;

    //std::mutex cropMutex;

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
     * @brief removeOutliers computes the centroid of the depth map
     *        wrt the image plane and removes (sets to 0) depth values
     *        of pixels with a distance > threshold from the centroid
     * @param image4d
     * @param threshold: max allowed distance
     */
    void removeOutliers(Image4D& image4d);
    void cropFaceThread(const std::vector<Image4D> &inputFaces, std::vector<Face>& croppedFaces, int begin, int end, std::mutex &cropMutex);
    void maskRGBToDepth(Image4D& image);
};

} // namespace face

#endif // FACE_PREPROCESSOR_H
