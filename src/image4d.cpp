#include "image4d.h"

#include <math.h>

using cv::Mat;

namespace face {

// ---------- constructors ----------

Image4D::Image4D() : WIDTH(0), HEIGHT(0), DEPTH_IMG_RATIO(0)
{
    image = Mat::zeros(1,1,CV_16UC3);
    depthMap = Mat::zeros(1,1,CV_16SC1);

    intrinsicMatrix = Mat::zeros(3, 3, CV_32FC1);
    intrinsicMatrix.at<float>(0,0) = 1;
    intrinsicMatrix.at<float>(1,1) = 1;
    intrinsicMatrix.at<float>(2,2) = 1;
    intrinsicMatrix.at<float>(0,2) = WIDTH/2;
    intrinsicMatrix.at<float>(1,2) = HEIGHT/2;
}


Image4D::Image4D(Mat &image, Mat &depthMap, const Mat &intrinsicCameraMatrix)
    : image(image), depthMap(depthMap)
{
    intrinsicCameraMatrix.copyTo(intrinsicMatrix);
    WIDTH  = depthMap.cols;
    HEIGHT = depthMap.rows;

    resizeImage();
}


// ---------- public member functions ----------

size_t Image4D::getWidth()  const { return WIDTH;  }
size_t Image4D::getHeight() const { return HEIGHT; }
size_t Image4D::getArea()   const { return WIDTH*HEIGHT;}
float  Image4D::getDepthImageRatio() const { return DEPTH_IMG_RATIO; }
Mat    Image4D::getIntrinsicMatrix() const
{
    Mat newIntrinsicMatrix;
    intrinsicMatrix.copyTo(newIntrinsicMatrix);
    return newIntrinsicMatrix;
}

Mat Image4D::get3DImage() const
{
    /*
    auto fx = intrinsicMatrix.at<float>(0,0);
    auto fy = intrinsicMatrix.at<float>(1,1);
    auto cx = intrinsicMatrix.at<float>(0,2);
    auto cy = intrinsicMatrix.at<float>(1,2);

    Mat image3D(HEIGHT, WIDTH, CV_32FC3);

    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            auto d = static_cast<float>(depthMap.at<uint16_t>(i,j));
            auto& vec = image3D.at<cv::Vec3f>(i,j);
            vec[0] = d * (float(j) - cx)/fx;
            vec[1] = d * (float(i) - cy)/fy;
            vec[2] = d;
        }
    }

    return image3D;
    */
    return cvtDepthMapTo3D(depthMap, intrinsicMatrix);
}


void Image4D::crop(const cv::Rect &cropRegion) {

    image    = image(cropRegion);    // crop image
    depthMap = depthMap(cropRegion); // crop depthMap

    WIDTH  = cropRegion.width;
    HEIGHT = cropRegion.height;

    intrinsicMatrix.at<float>(0,2) -= cropRegion.x;
    intrinsicMatrix.at<float>(1,2) -= cropRegion.y;
}


// ---------- private member functions ----------

void Image4D::resizeImage()
{
    const int IMG_WIDTH  = image.cols;
    const int IMG_HEIGHT = image.rows;

    DEPTH_IMG_RATIO = static_cast<float>(depthMap.cols) / IMG_WIDTH;

    if (DEPTH_IMG_RATIO == 1)
        return;

    assert(static_cast<float>(depthMap.rows) / IMG_HEIGHT == DEPTH_IMG_RATIO &&
            "Image and cloud sizes are not proportional!");

    assert(DEPTH_IMG_RATIO < 1 && "Image is smaller than cloud!");

    cv::Size newImageSize(IMG_WIDTH * DEPTH_IMG_RATIO, IMG_HEIGHT * DEPTH_IMG_RATIO);
    cv::resize(image, image, newImageSize, cv::INTER_AREA);

    intrinsicMatrix.at<float>(0,2) *= DEPTH_IMG_RATIO;
    intrinsicMatrix.at<float>(1,2) *= DEPTH_IMG_RATIO;

    return;
}


Mat cvtDepthMapTo3D(const Mat& depthMap, const Mat& intrinsicMatrix) {
    auto fx = intrinsicMatrix.at<float>(0,0);
    auto fy = intrinsicMatrix.at<float>(1,1);
    auto cx = intrinsicMatrix.at<float>(0,2);
    auto cy = intrinsicMatrix.at<float>(1,2);

    const auto HEIGHT = depthMap.rows;
    const auto WIDTH  = depthMap.cols;
    Mat image3D(HEIGHT, WIDTH, CV_32FC3);

    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            auto d = static_cast<float>(depthMap.at<uint16_t>(i,j));
            auto& vec = image3D.at<cv::Vec3f>(i,j);
            vec[0] = d * (float(j) - cx)/fx;
            vec[1] = d * (float(i) - cy)/fy;
            vec[2] = d;
        }
    }

    return image3D;
}

cv::Mat cvt3DToDepthMap(const cv::Mat& image3D,  const cv::Mat& intrinsicMatrix) {

    auto fx = intrinsicMatrix.at<float>(0,0);
    auto fy = intrinsicMatrix.at<float>(1,1);
    auto cx = intrinsicMatrix.at<float>(0,2);
    auto cy = intrinsicMatrix.at<float>(1,2);

    const auto HEIGHT = image3D.rows;
    const auto WIDTH  = image3D.cols;
    Mat depthMap = Mat::zeros(HEIGHT, WIDTH, CV_16SC1);

    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            auto &vec = image3D.at<cv::Vec3f>(i,j);
            auto x = fx*vec[0]/vec[2] + cx;
            auto y = fy*vec[1]/vec[2] + cy;

            if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)
                depthMap.at<uint16_t>(y,x) = vec[2];
        }
    }

    return depthMap;
}

}
