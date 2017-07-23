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
    depthMap = Mat::zeros(1,1,CV_16SC1);
}

Face::Face(cv::Mat &image, cv::Mat &depthMap) : image(image), depthMap(depthMap)
{
    WIDTH  = depthMap.cols;
    HEIGHT = depthMap.rows;

    resizeImage();
}

Face::Face(cv::Mat &image, cv::Mat &depthMap, const cv::Mat &intrinsicCameraMatrix)
    : image(image), depthMap(depthMap), intrinsicMatrix(intrinsicCameraMatrix) {

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
    float fx = float(intrinsicCameraMatrix.at<double>(0,0));
    float fy = float(intrinsicCameraMatrix.at<double>(1,1));
    float cx = float(intrinsicCameraMatrix.at<double>(0,2));
    float cy = float(intrinsicCameraMatrix.at<double>(1,2));

    Mat image3D(HEIGHT, WIDTH, CV_32FC3);
    //std::vector<cv::Point3f> objectPoints;

    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            float d = static_cast<float>(depthMap.at<uint16_t>(i,j));// - 600;
            auto& vec = image3D.at<cv::Vec3f>(i,j);
            vec[0] = d * (float(j) - cx)/fx;
            vec[1] = d * (float(i) - cy)/fy;
            vec[2] = d;

            std::cout << "Point: (" << vec[0] << "," << vec[1] << "," << vec[2] << ")" << std::endl;
           //objectPoints.emplace_back(vec[0], vec[1], vec[2]);
        }
    }
/*
    std::vector<float> rvec;
    cv::Rodrigues(Mat::eye(3,3, CV_32FC1), rvec);

    std::vector<float> tvec = {0,0,800};
    std::vector<cv::Point2f> imagePoints;
    projectPoints(objectPoints, rvec, tvec, intrinsicCameraMatrix, std::vector<float>(), imagePoints);

    Mat image3D2(HEIGHT, WIDTH, CV_32FC3);
    image3D2.setTo(0);
    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            auto& projPoint = imagePoints[i*WIDTH + j];
            std::cout << "point: (" << projPoint.x << "," << projPoint.y << ")" << std::endl;
            if (projPoint.x > 0 && projPoint.x < WIDTH && projPoint.y > 0 && projPoint.y < HEIGHT) {
                auto& point = objectPoints[i*WIDTH +j];
                image3D2.at<cv::Vec3f>(projPoint.y,projPoint.x) = {point.x, point.y, point.z};
            }
            else {
                std::cout << "Skipping point" << std::endl;
            }
        }
    }*/

    return image3D;
}


void Face::crop(const cv::Rect &cropRegion) {

    image    = image(cropRegion);    // crop image
    depthMap = depthMap(cropRegion); // crop cloud

    WIDTH  = cropRegion.width;
    HEIGHT = cropRegion.height;

    intrinsicMatrix.at<double>(0,2) -= cropRegion.x;
    intrinsicMatrix.at<double>(1,2) -= cropRegion.y;
}

void Face::depthForEach(std::function<void(int, int, uint16_t&)> function, const cv::Rect& ROI) {

    const uint MAX_X = ROI.x + ROI.height;
    const uint MAX_Y = ROI.y + ROI.width;

    for (uint x = ROI.x; x < MAX_X; ++x) {
        for (uint y = ROI.y; y < MAX_Y; ++y) {
            function(x,y,depthMap.at<uint16_t>(x,y));
        }
    }
}

void Face::depthForEach(std::function<void(int, int, const uint16_t &)> function, const cv::Rect& ROI)  const {

    const uint MAX_X = ROI.x + ROI.height;
    const uint MAX_Y = ROI.y + ROI.width;

    for (uint x = ROI.x; x < MAX_X; ++x) {
        for (uint y = ROI.y; y < MAX_Y; ++y) {
            function(x,y,depthMap.at<uint16_t>(x,y));
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

    assert(static_cast<float>(depthMap.rows) / IMG_HEIGHT == DEPTH_IMG_RATIO &&
            "Image and cloud sizes are not proportional!");

    assert(DEPTH_IMG_RATIO < 1 && "Image is smaller than cloud!");

    cv::Size newImageSize(IMG_WIDTH * DEPTH_IMG_RATIO, IMG_HEIGHT * DEPTH_IMG_RATIO);
    cv::resize(image, image, newImageSize, cv::INTER_AREA);

    intrinsicMatrix.at<double>(0,2) *= DEPTH_IMG_RATIO;
    intrinsicMatrix.at<double>(1,2) *= DEPTH_IMG_RATIO;

    return;
}
