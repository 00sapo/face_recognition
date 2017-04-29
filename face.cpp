#include "face.h"

#include <math.h>
#include <opencv2/core/eigen.hpp>

#include "pointprojector.h"
#include "singletonsettings.h"

using std::isnan;
using cv::Mat;
using pcl::PointCloud;
using pcl::PointXYZ;

Face::Face()
{
    image = Mat::zeros(1,1,CV_16UC3);
    cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
}

Face::Face(Mat image, PointCloud<PointXYZ>::Ptr cloud) : image(image), cloud(cloud) {
    cloudImageRatio = image.cols/cloud->width;
}

Face::Face(Mat image, PointCloud<PointXYZ>::Ptr cloud, float cloudImageRatio) :
    image(image), cloud(cloud), cloudImageRatio(cloudImageRatio) { }

Mat Face::get3DImage()
{
    Mat image3D(cv::Size(image.cols, image.rows), CV_32FC3);

    // TODO: probably NANs should be removed unless we guarantee they can't be present
    //for (int x = 0; x < cloud->width; ++x) {
     //   for (int y = 0; y < cloud->height; ++y) {
    for (unsigned int i = 0; i < cloud->size(); ++i) {
        int x = i / cloud->width;
        int y = i % cloud->width / 4;
        const PointXYZ& point = cloud->at(i);
        image3D.at<Vec3f>(x,y) = {point.x, point.y, point.z};
    }

    Mat displayDepth = Mat::zeros(cv::Size(image.cols, image.rows), CV_32FC1);
    for (int x = 0; x < image.cols; ++x) {
        for (int y = 0; y < image.rows; ++y) {
            float depth = image3D.at<Vec3f>(x,y)[2];
            if(std::isnan(depth) || std::isinf(depth))
                depth = 1000;
            displayDepth.at<float>(x,y) = depth;
        }
    }

    Mat newMat;
    cv::normalize(displayDepth, newMat, 0, 255, NORM_MINMAX, CV_8UC1);

    imshow("Depth", newMat);
    waitKey(0);


    /*
    Mat mask = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);
    for(auto& point : *cloud) {
        if(!isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
            auto coord = get2DCoordinates(point);
            if(coord.x < 0 || coord.x > image.rows || coord.y < 0 || coord.y > image.cols) {
                std::cout << "coord.x: " << coord.x << std::endl;
                std::cout << "coord.y: " << coord.y << std::endl;
                depthMap.at<float>(coord.x, coord.y) = FLT_MAX;
            }
            else {
              depthMap.at<float>(coord.x, coord.y) = point.z;
              mask.at<char>(coord.x, coord.y) = 255;
            }
        }
    }

    Mat normalizedDepthMap;
    */
    //cv::normalize(depthMap, normalizedDepthMap, 0, 255, NORM_MINMAX, CV_8UC1, mask);

    return image3D;
}

Point2i Face::get2DCoordinates(const pcl::PointXYZ &point) const
{

    // Define projection matrix from 3D to 2D:
    //P matrix is in camera_info.yaml
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> P;
    Mat matP = SingletonSettings::getInstance().getP();

    cv2eigen(matP, P);
    P(0,0) *= cloudImageRatio;
    P(0,2) *= cloudImageRatio;
    P(1,1) *= cloudImageRatio;
    P(1,2) *= cloudImageRatio;

    // 3D to 2D projection:
    //Let's do P*point and rescale X,Y
    Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1);
    Eigen::Vector3f output = P * homogeneous_point;
    output[0] /= output[2];
    output[1] /= output[2];

    return Point2i((unsigned int)output[1], (unsigned int)output[0]);
}
