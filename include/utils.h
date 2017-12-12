#ifndef FACE_LBP_H
#define FACE_LBP_H

#include "image4d.h"

#include <bitset>
#include <opencv2/opencv.hpp>
#include <thread>

namespace face {

/**
 * @brief OLBP OLBPHist compute the uniform LBP of a source Mat with ray=1 and sampling=8
 * @param src
 * @return a Mat containing the LBP values
 */
template <typename _Tp>
cv::Mat OLBP_(const cv::Mat src)
{
    cv::Mat dst = cv::Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            _Tp center = src.at<_Tp>(i, j);
            std::bitset<8> code(0);
            code |= (src.at<_Tp>(i - 1, j - 1) > center) << 7;
            code |= (src.at<_Tp>(i - 1, j) > center) << 6;
            code |= (src.at<_Tp>(i - 1, j + 1) > center) << 5;
            code |= (src.at<_Tp>(i, j + 1) > center) << 4;
            code |= (src.at<_Tp>(i + 1, j + 1) > center) << 3;
            code |= (src.at<_Tp>(i + 1, j) > center) << 2;
            code |= (src.at<_Tp>(i + 1, j - 1) > center) << 1;
            code |= (src.at<_Tp>(i, j - 1) > center) << 0;
            // xor marks bits that are not the same as their neighbors on the left
            code ^= (code >> 1);
            // count number of set bits on code and subtract 1 if bit shifted in previous expression was 1
            int ntransitions = code.count() - code[code.size() - 1];
            // if the number of transiotions is > 2 (not uniform) use 9 as default value (the minimum value not uniform available)
            unsigned char value = (unsigned char)ntransitions > 2 ? 9 : code.to_ulong();

            dst.at<unsigned char>(i - 1, j - 1) = value;
        }
    }

    return dst;
}

inline cv::Mat OLBP(const cv::Mat src)
{
    cv::Mat lbp;
    switch (src.type()) {
    case CV_8SC1:
        lbp = OLBP_<char>(src);
        break;
    case CV_8UC1:
        lbp = OLBP_<unsigned char>(src);
        break;
    case CV_16SC1:
        lbp = OLBP_<short>(src);
        break;
    case CV_16UC1:
        lbp = OLBP_<unsigned short>(src);
        break;
    case CV_32SC1:
        lbp = OLBP_<int>(src);
        break;
    case CV_32FC1:
        lbp = OLBP_<float>(src);
        break;
    case CV_64FC1:
        lbp = OLBP_<double>(src);
        break;
    default:
        break;
    }
    return lbp;
}

/**
 * @brief OLBPHist compute the uniform LBP histogram of a source Mat with ray=1 and sampling=8
 * @param src
 * @return a Mat with 1 dimension containing the LBP histogram, consisting in 59 integer values
 */
inline cv::Mat OLBPHist(const cv::Mat src)
{
    cv::Mat hist;
    cv::Mat lbp = OLBP(src);

    int histSize = 59;
    float range[] = { 0, 60 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::calcHist(&lbp, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    return hist;
}

/**
 * @brief HistMean_ computes the mean of a histogram
 * @param hist a Mat containing a histogram (i.e. returned by cv::calcHist())
 * @return a float representing the mean
 */
template <typename _Tp>
float HistMean_(const cv::Mat hist)
{
    float mean = 0;
    for (int i = 0; i < hist.cols; i++) {
        for (int j = 0; j < hist.rows; j++) {
            mean += hist.at<_Tp>(i, j);
        }
    }

    return mean / (hist.cols + hist.rows);
}

inline float HistMean(const cv::Mat src)
{
    float mean;
    switch (src.type()) {
    case CV_8SC1:
        mean = HistMean_<char>(src);
        break;
    case CV_8UC1:
        mean = HistMean_<unsigned char>(src);
        break;
    case CV_16SC1:
        mean = HistMean_<short>(src);
        break;
    case CV_16UC1:
        mean = HistMean_<unsigned short>(src);
        break;
    case CV_32SC1:
        mean = HistMean_<int>(src);
        break;
    case CV_32FC1:
        mean = HistMean_<float>(src);
        break;
    case CV_64FC1:
        mean = HistMean_<double>(src);
        break;
    default:
        break;
    }
    return mean;
}

template <class Fn, class... Args>
/**
 * @brief multiThreadVectorProcessing takes a function and a vector of images and executes it in multiple threads
 * @param images the vector of images
 * @param function the function to execute. N.B. the @Image4D argument MUST be the last one
 * @param args the arguments of the function, as you would pass to &std::thread constructor except of the @Image4D argument
 */
void multiThreadVectorProcessing(std::vector<Image4D> images, Fn function, Args... args)
{
    int n_proc = std::thread::hardware_concurrency();
    std::thread threads[n_proc];
    int i = 0;

    for (auto& face : images) {
        if (i >= n_proc) {
            for (auto& t : threads)
                if (t.joinable())
                    t.join();
            i = 0;
        }
        threads[i++] = std::thread(function, args..., std::ref(face));
    }
    for (auto& t : threads)
        if (t.joinable())
            t.join();
}
}

#endif // FACE_LBP_H
