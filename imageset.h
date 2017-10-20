#ifndef IMAGESET_H
#define IMAGESET_H
#include <boost/any.hpp>
#include <opencv2/core.hpp>

namespace face {

class Image4DSet {
public:
    Image4DSet();

    template <typename T>
    void depthForEach(const std::function<void(int, int, T&)>& function, const cv::Rect& ROI)
    {
        return boost::any_cast<T>(virtualDepthForEach(function, ROI));
    }

    template <typename T>
    void imageForEach(const std::function<void(int, int, T&)>& function, const cv::Rect& ROI)
    {
        return boost::any_cast<T>(virtualImageForEach(function, ROI));
    }

    virtual cv::Mat const* getImage() const = 0;

    virtual cv::Mat const* getDepthMap() const = 0;

protected:
    virtual boost::any virtualDepthForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI) = 0;
    virtual boost::any virtualImageForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI) = 0;
};
}
#endif // IMAGESET_H
