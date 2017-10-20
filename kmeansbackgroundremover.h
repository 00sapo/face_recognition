#ifndef KMEANSBACKGROUNDREMOVER_H
#define KMEANSBACKGROUNDREMOVER_H
#include "filter.h"

namespace face {

class KmeansBackgroundRemover : public Filter {
public:
    KmeansBackgroundRemover(uint16_t fixedThreshold = 1600);

    /**
     * @brief filter remove the background on the first 4D image of the set using kmeans
     * @param image
     * @return
     */
    bool filter(face::Image4DSet& image);

private:
    bool faceDetectorAvailable = false;
    cv::CascadeClassifier classifier;
    uint16_t fixedThreshold = 1600;

    void removeBackgroundFixed(Image4DSet& face, uint16_t threshold) const;
    void removeBackgroundDynamic(Image4DSet& face, const cv::Rect& boundingBox) const;
    bool detectForegroundFace(const Image4DSet& face, cv::Rect& boundingBox);
};
}
#endif // KMEANSBACKGROUNDREMOVER_H
