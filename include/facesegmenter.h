#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H

#include <opencv2/objdetect.hpp>

#include "face.h"

/**
 * @brief The BackgroundSegmentation class performs the preprocessing
 */
class FaceSegmenter {
public:
    /**
     * @brief BackgroundSegmentation: constructor
     */
    FaceSegmenter();

    /**
     * @brief BackgroundSegmentation: constructor
     */
    FaceSegmenter(const std::string& faceDetectorPath);

    /**
     * @brief detectForegroundFace detects the nearest face in the image
     * @param face: Face containing the image
     * @param detectedFace: ROI of the detected face
     * @return false if no face was detected, true otherwise
     */
    bool detectForegroundFace(const Face& face, const cv::Size &outputSize, cv::Rect& detectedRegion);

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

private:
    static const std::string FACE_DETECTOR_PATH;

    bool faceDetectorAvailable = false;

    cv::CascadeClassifier classifier;
};

#endif // BACKGROUNDSEGMENTATION_H
