#include "face.h"

#include <math.h>

using std::isnan;
using cv::Mat;
using pcl::PointCloud;
using pcl::PointXYZ;

// ---------- constructors ----------

Face::Face()
{
    image = Mat::zeros(1,1,CV_16UC3);
    cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
}

Face::Face(Mat image, PointCloud<PointXYZ>::Ptr cloud) : image(image), cloud(cloud)
{
    cloudImageRatio = image.cols/cloud->width;
    WIDTH  = cloud->width;
    HEIGHT = cloud->height;
}

Face::Face(Mat image, PointCloud<PointXYZ>::Ptr cloud, float cloudImageRatio)
    : image(image), cloud(cloud), cloudImageRatio(cloudImageRatio)
{
    WIDTH  = cloud->width;
    HEIGHT = cloud->height;
}

// ---------- public member functions ----------

uint Face::getWidth()  const { return WIDTH;  }
uint Face::getHeight() const { return HEIGHT; }

Mat Face::get3DImage() const
{
    Mat image3D(cv::Size(WIDTH, HEIGHT), CV_32FC3);

   for (uint i = 0; i < cloud->size(); ++i) {
        int y = i / WIDTH;
        int x = i % WIDTH / 4;
        const PointXYZ& point = cloud->at(i);
        image3D.at<cv::Vec3f>(x,y) = {point.x, point.y, point.z};
    }

    Mat displayDepth = Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32FC1);
    for (uint x = 0; x < HEIGHT; ++x) {
        for (uint y = 0; y < WIDTH; ++y) {
            float depth = image3D.at<cv::Vec3f>(x,y)[2];
            if(!std::isnan(depth) && !std::isinf(depth))
                displayDepth.at<float>(x,y) = depth;
        }
    }

    Mat newMat;
    cv::normalize(displayDepth, newMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    imshow("Depth", newMat);
    cv::waitKey(0);

    return image3D;
}
