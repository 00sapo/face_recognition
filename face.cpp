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

Mat Face::get3DImage()
{
    Mat image3D(cv::Size(WIDTH, HEIGHT), CV_32FC3);

    cloudForEach([image3D](unsigned int x, unsigned int y, PointXYZ& point) mutable {
        image3D.at<cv::Vec3f>(x,y) = {point.x, point.y, point.z};
     });

    return image3D;
}


void Face::cloudForEach(std::function<void(uint, uint, pcl::PointXYZ &)> function) {
    const auto SIZE = cloud->size();
    for (ulong i = 0; i < SIZE; ++i) {
        int x = i / WIDTH;
        int y = i % WIDTH;
        function(x,y,cloud->at(i));
    }
}

void Face::imageForEach(std::function<void(uint,uint,float&)> function) {
    for (uint x = 0; x < HEIGHT; ++x) {
        for (uint y = 0; y < WIDTH; ++y) {
            function(x,y,image.at<float>(x,y));
        }
    }
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
