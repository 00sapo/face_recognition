#include "face.h"

#include <math.h>

using std::cout;
using std::endl;

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

    cloudForEach([image3D](int x, int y, PointXYZ& point) mutable {
        image3D.at<cv::Vec3f>(x,y) = {point.x, point.y, point.z};
     });

    return image3D;
}


void Face::crop(const cv::Rect &cropRegion) {

    // crop image
    cout << "Cropping image..." << endl;
    image = image(cropRegion);
    cout << "Done!" << endl;

    // crop cloud
    cout << "Cropping cloud..." << endl;
    PointCloud<PointXYZ>::Ptr croppedCloud(new PointCloud<PointXYZ>(cropRegion.width, cropRegion.height));
    auto lambda = [croppedCloud, cropRegion](int x, int y, PointXYZ& point) mutable {
                        croppedCloud->at(x-cropRegion.x, y-cropRegion.y) = point;
                  };

    cloudForEach(lambda, cropRegion);
    cloud = croppedCloud;
    cout << "Done!" << endl;

    WIDTH  = cropRegion.width;
    HEIGHT = cropRegion.height;
}


void Face::cloudForEach(std::function<void(int, int, pcl::PointXYZ &)> function) {

    // if cloud is organized I can skip the computation of x and y at every iteration
    if (cloud->isOrganized()) {
        for (uint x = 0; x < HEIGHT; ++x) {
            for (uint y = 0; y < WIDTH; ++y) {
                function(x,y,cloud->at(x,y));
            }
        }
    }

    // otherwise I must compute x and y at every iteration
    else {
        const auto SIZE = cloud->size();
        for (ulong i = 0; i < SIZE; ++i) {
            int x = i / WIDTH;
            int y = i % WIDTH;
            function(x,y,cloud->at(i));
        }
    }
}

void Face::cloudForEach(std::function<void(int, int, pcl::PointXYZ &)> function, const cv::Rect& ROI) {

    const uint MAX_X = ROI.x + ROI.height;
    const uint MAX_Y = ROI.y + ROI.width;

    // if cloud is organized I can skip the computation of i at every iteration
    if (cloud->isOrganized()) {
        for (uint x = ROI.x; x < MAX_X; ++x) {
            for (uint y = ROI.y; y < MAX_Y; ++y) {
                function(x,y,cloud->at(x,y));
            }
        }
    }

    // otherwise I must compute i at every iteration
    else {
        for (uint x = ROI.x; x < MAX_X; ++x) {
            for (uint y = ROI.y; y < MAX_Y; ++y) {
                uint i = x * WIDTH + y;
                function(x,y,cloud->at(i));
            }
        }
    }
}

void Face::imageForEach(std::function<void(int, int, float &)> function) {
    for (uint x = 0; x < HEIGHT; ++x) {
        for (uint y = 0; y < WIDTH; ++y) {
            function(x,y,image.at<float>(x,y));
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
