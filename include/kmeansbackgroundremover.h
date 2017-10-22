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
    bool filter();

    Image4DComponent* getImage4DComponent() const;
    void setImage4DComponent(Image4DComponent* value);

    uint16_t getFixedThreshold() const;
    void setFixedThreshold(const uint16_t& value);

    cv::CascadeClassifier getClassifier() const;
    void setClassifier(const cv::CascadeClassifier& value);

    bool isFaceDetectorAvailable() const;

private:
    bool faceDetectorAvailable = false;
    cv::CascadeClassifier classifier;
    uint16_t fixedThreshold;
    Image4DComponent* image4d;

    void removeBackgroundFixed() const;
    void removeBackgroundDynamic(cv::Rect& boundingBox) const;
    bool detectForegroundFace(cv::Rect& boundingBox);
    static void filterImage4DComponent(Image4DComponent* image4d);
};
}
#endif // KMEANSBACKGROUNDREMOVER_H
