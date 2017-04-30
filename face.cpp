#include "face.h"

#include <math.h>

using std::isnan;
using cv::Mat;
using pcl::PointCloud;
using pcl::PointXYZ;

// ---------- constructors ----------

Face::Face() : WIDTH(0), HEIGHT(0), CLOUD_IMG_RATIO(0)
{
    image = Mat::zeros(1,1,CV_16UC3);
    cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
}

Face::Face(Mat image, PointCloud<PointXYZ>::Ptr cloud) : image(image), cloud(cloud)
{
    WIDTH  = cloud->width;
    HEIGHT = cloud->height;

    resizeImage();
}

// ---------- public member functions ----------

uint  Face::getWidth()  const { return WIDTH;  }
uint  Face::getHeight() const { return HEIGHT; }
float Face::getCloudImageRatio() const { return CLOUD_IMG_RATIO; }

Mat Face::get3DImage() const
{
    Mat image3D(cv::Size(WIDTH, HEIGHT), CV_32FC3);

    Mat displayDepth = Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32FC1);

    auto size = cloud->size();
    for (uint i = 0; i < cloud->size(); ++i) {
        int x = i / WIDTH;
        int y = i % WIDTH;
        const PointXYZ& point = cloud->at(i);
        //image3D.at<cv::Vec3f>(x,y) = {point.x, point.y, point.z};
        displayDepth.at<float>(x,y) = point.z;
    }

    cv::normalize(displayDepth, displayDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    std::cout << "newMat width: "    << displayDepth.cols
              << "\nnewMat height: " << displayDepth.rows << std::endl;

    imshow("Depth", displayDepth);
    cv::waitKey(0);

    return image3D;
}

// ---------- private member functions ----------

void Face::resizeImage()
{
    const int IMG_WIDTH  = image.cols;
    const int IMG_HEIGHT = image.rows;

    CLOUD_IMG_RATIO = static_cast<float>(cloud->width) / IMG_WIDTH;

    if (CLOUD_IMG_RATIO == 1)
        return;

    assert (static_cast<float>(cloud->height) / IMG_HEIGHT == CLOUD_IMG_RATIO &&
            "Asssert failed: image and cloud sizes are not proportional!");

    assert (CLOUD_IMG_RATIO < 1 &&
            "Assert failed: image is smaller than cloud!");

    cv::Size newImageSize(IMG_WIDTH * CLOUD_IMG_RATIO, IMG_HEIGHT * CLOUD_IMG_RATIO);
    cv::resize(image, image, newImageSize, cv::INTER_AREA);

    return;
}
