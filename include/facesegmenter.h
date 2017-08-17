#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H

#include <opencv2/objdetect.hpp>

namespace face {


class Image4D;

/**
 * @brief The BackgroundSegmentation class performs the preprocessing
 */
class FaceSegmenter {
public:

    /**
     * @brief BackgroundSegmentation: constructor
     */
    explicit FaceSegmenter(const std::string& faceDetectorPath = FACE_DETECTOR_PATH);

    /**
     * @brief Preprocesses the image removing the backround
     * @param faces: vector of faces to preprocess
     */

    bool segment(Image4D &face, cv::Rect &faceRegion);
    bool segment(std::vector<Image4D>& faces, std::vector<cv::Rect> &faceRegions);

private:
    static const std::string FACE_DETECTOR_PATH;

    bool faceDetectorAvailable = false;

    cv::CascadeClassifier classifier;

    /**
     * @brief detectForegroundFace detects the nearest face in the image
     * @param face: Face containing the image
     * @param detectedFace: ROI of the detected face
     * @return false if no face was detected, true otherwise
     */
    bool detectForegroundFace(const Image4D& face, cv::Rect &boundingBox);

    /**
     * @brief removeBackground splits the face cloud in two clusters, discarding
     *        furthest one
     * @param face
     */
    bool removeBackgroundDynamic(Image4D& face, const cv::Rect &boundingBox) const;

    bool removeBackgroundFixed(Image4D& face, uint16_t threshold) const;
};

}   // face

#endif // BACKGROUNDSEGMENTATION_H
