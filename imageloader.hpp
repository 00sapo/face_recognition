#ifndef IMAGELOADER_H
#define IMAGELOADER_H

#include "global_includes.hpp"
#include <opencv2/opencv.hpp>


/**
 * @brief The ImageLoader class extends VideoCapture class adding
 *        the possibility to load an image sequence from multiple
 *        files in a vector
 */
class ImageLoader : public cv::VideoCapture
{
public:

    float downscalingRatio = 1;

    ImageLoader();
    ImageLoader(const std::string &path);

    bool get(cv::Mat& image);
    bool get(std::vector<cv::Mat> &imageSequence);
    bool get(const std::string &path, cv::Mat &image);
    bool get(const std::string &path, std::vector<cv::Mat> &imageSequence);
};

#endif // IMAGELOADER_H
