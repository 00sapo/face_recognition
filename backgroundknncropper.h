#ifndef BACKGROUNDSUBTRACTOR_H
#define BACKGROUNDSUBTRACTOR_H
#include <filter.h>
#include <opencv2/bgsegm.hpp>
#include <opencv2/video.hpp>

namespace face {
/**
 * @brief The BackgroundSubtractor class implements a background subtractor based on image comparison
 */
class BackgroundKNNCropper : public Filter {
public:
    std::string actionToPrint() { return "Removing background using KNN algorithm..."; }
    BackgroundKNNCropper();

    bool filter();
    Image4DComponent* getImage4DComponent() const;
    void setImage4DComponent(Image4DComponent* value);

protected:
    void cropImage(cv::Mat& maskKNN, cv::Rect& roi);

private:
    cv::Ptr<cv::BackgroundSubtractor> subtractor;
    Image4DComponent* imageSet;
};
}
#endif // BACKGROUNDSUBTRACTOR_H
