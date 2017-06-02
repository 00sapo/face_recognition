#include "face.h"

#include <math.h>

using std::cout;
using std::endl;

using std::isnan;
using cv::Mat;
using pcl::PointCloud;
using pcl::PointXYZ;

// ---------- constructors ----------

Face::Face() : WIDTH(0), HEIGHT(0), DEPTH_IMG_RATIO(0)
{
    image = Mat::zeros(1,1,CV_16UC3);
    depthMap = Mat::zeros(1,1,CV_16U);
}

Face::Face(Mat image, cv::Mat depthMap) : image(image), depthMap(depthMap)
{
    WIDTH  = depthMap.cols;
    HEIGHT = depthMap.rows;

    resizeImage();
}

// ---------- public member functions ----------

size_t Face::getWidth()  const { return WIDTH;  }
size_t Face::getHeight() const { return HEIGHT; }
size_t Face::getArea() const {return WIDTH*HEIGHT;}
float Face::getDepthImageRatio() const { return DEPTH_IMG_RATIO; }

Mat Face::get3DImage(const Mat& intrinsicCameraMatrix) const
{
    float fx = intrinsicCameraMatrix.at<float>(0,0);
    float fy = intrinsicCameraMatrix.at<float>(1,1);
    float cx = intrinsicCameraMatrix.at<float>(0,2);
    float cy = intrinsicCameraMatrix.at<float>(1,2);

    Mat image3D(HEIGHT, WIDTH, CV_32FC3);

    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            float d = depthMap.at<float>(i,j);
            auto& vec = image3D.at<cv::Vec3f>(i,j);
            vec[0] = d * (float(j) - cx)/fx;
            vec[1] = d * (float(i) - cy)/fy;
            vec[2] = d;
        }
    }

    return image3D;
}


void Face::crop(const cv::Rect &cropRegion) {

    image    = image(cropRegion);    // crop image
    depthMap = depthMap(cropRegion); // crop cloud

    WIDTH  = cropRegion.width;
    HEIGHT = cropRegion.height;
}

void Face::depthForEach(std::function<void(int, int, float&)> function, const cv::Rect& ROI) {

    const uint MAX_X = ROI.x + ROI.height;
    const uint MAX_Y = ROI.y + ROI.width;

    for (uint x = ROI.x; x < MAX_X; ++x) {
        for (uint y = ROI.y; y < MAX_Y; ++y) {
            function(x,y,depthMap.at<float>(x,y));
        }
    }
}

void Face::depthForEach(std::function<void(int, int, const float&)> function, const cv::Rect& ROI)  const {

    const uint MAX_X = ROI.x + ROI.height;
    const uint MAX_Y = ROI.y + ROI.width;

    for (uint x = ROI.x; x < MAX_X; ++x) {
        for (uint y = ROI.y; y < MAX_Y; ++y) {
            function(x,y,depthMap.at<float>(x,y));
        }
    }
}

void Face::imageForEach(std::function<void(int, int, float&)> function, const cv::Rect& ROI) {
    const uint MAX_X = ROI.x + ROI.height;
    const uint MAX_Y = ROI.y + ROI.width;
    for (uint x = ROI.x; x < MAX_X; ++x) {
        for (uint y = ROI.y; y < MAX_Y; ++y) {
            function(x,y,image.at<float>(x,y));
        }
    }
}

void Face::imageForEach(std::function<void(int, int, const float&)> function, const cv::Rect& ROI) const {
    const uint MAX_X = ROI.x + ROI.height;
    const uint MAX_Y = ROI.y + ROI.width;
    for (uint x = ROI.x; x < MAX_X; ++x) {
        for (uint y = ROI.y; y < MAX_Y; ++y) {
            function(x,y,image.at<float>(x,y));
        }
    }
}


// ---------- private member functions ----------

void Face::resizeImage()
{
    const int IMG_WIDTH  = image.cols;
    const int IMG_HEIGHT = image.rows;

    DEPTH_IMG_RATIO = static_cast<float>(depthMap.cols) / IMG_WIDTH;

    if (DEPTH_IMG_RATIO == 1)
        return;

    assert (static_cast<float>(depthMap.rows) / IMG_HEIGHT == DEPTH_IMG_RATIO &&
            "Asssert failed: image and cloud sizes are not proportional!");

    assert (DEPTH_IMG_RATIO < 1 &&
            "Assert failed: image is smaller than cloud!");

    cv::Size newImageSize(IMG_WIDTH * DEPTH_IMG_RATIO, IMG_HEIGHT * DEPTH_IMG_RATIO);
    cv::resize(image, image, newImageSize, cv::INTER_AREA);

    return;
}
