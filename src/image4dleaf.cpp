#include "image4dleaf.h"

#include <image4dvectorcomposite.h>
#include <math.h>

using cv::Mat;

namespace face {

// ---------- constructors ----------

Image4DLeaf::Image4DLeaf()
    : WIDTH(0)
    , HEIGHT(0)
    , DEPTH_IMG_RATIO(0)
{
    image = Mat::zeros(1, 1, CV_16UC3);
    depthMap = Mat::zeros(1, 1, CV_16SC1);

    intrinsicMatrix = Mat::zeros(3, 3, CV_64FC1);
    intrinsicMatrix.at<double>(0, 0) = 1;
    intrinsicMatrix.at<double>(1, 1) = 1;
    intrinsicMatrix.at<double>(2, 2) = 1;
    intrinsicMatrix.at<double>(0, 2) = WIDTH / 2;
    intrinsicMatrix.at<double>(1, 2) = HEIGHT / 2;
}

Image4DLeaf::Image4DLeaf(Mat& image, Mat& depthMap, const Mat& intrinsicCameraMatrix)
    : image(image)
    , depthMap(depthMap)
{
    intrinsicCameraMatrix.copyTo(intrinsicMatrix);
    WIDTH = depthMap.cols;
    HEIGHT = depthMap.rows;

    resizeImage();
}

// ---------- public member functions ----------

size_t Image4DLeaf::getWidth() const { return WIDTH; }
size_t Image4DLeaf::getHeight() const { return HEIGHT; }
size_t Image4DLeaf::getArea() const { return WIDTH * HEIGHT; }
float Image4DLeaf::getDepthImageRatio() const { return DEPTH_IMG_RATIO; }
Mat Image4DLeaf::getIntrinsicMatrix() const
{
    Mat newIntrinsicMatrix;
    intrinsicMatrix.copyTo(newIntrinsicMatrix);
    return newIntrinsicMatrix;
}

Mat Image4DLeaf::get3DImage() const
{
    float fx = float(intrinsicMatrix.at<double>(0, 0));
    float fy = float(intrinsicMatrix.at<double>(1, 1));
    float cx = float(intrinsicMatrix.at<double>(0, 2));
    float cy = float(intrinsicMatrix.at<double>(1, 2));

    Mat image3D(HEIGHT, WIDTH, CV_32FC3);

    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            float d = static_cast<float>(depthMap.at<uint16_t>(i, j));
            auto& vec = image3D.at<cv::Vec3f>(i, j);
            vec[0] = d * (float(j) - cx) / fx;
            vec[1] = d * (float(i) - cy) / fy;
            vec[2] = d;
        }
    }

    return image3D;
}

void Image4DLeaf::crop(const cv::Rect& cropRegion)
{

    image = image(cropRegion); // crop image
    depthMap = depthMap(cropRegion); // crop depthMap

    WIDTH = cropRegion.width;
    HEIGHT = cropRegion.height;

    intrinsicMatrix.at<double>(0, 2) -= cropRegion.x;
    intrinsicMatrix.at<double>(1, 2) -= cropRegion.y;
}

std::vector<Pose> Image4DLeaf::getRotationMatrix() const
{
    // Calculate rotation around x axis
    float cosx = cos(eulerAngles[0]);
    float senx = sin(eulerAngles[0]);
    float cosy = cos(eulerAngles[1]);
    float seny = sin(eulerAngles[1]);
    float cosz = cos(eulerAngles[2]);
    float senz = sin(eulerAngles[2]);

    return std::vector<Pose>({ Pose(cosy * cosz, cosx * senz + senx * seny * cosz, senx * senz - cosx * seny * cosz,
        -cosy * senz, cosx * cosz - senx * seny * senz, senx * cosz + cosx * seny * senz,
        seny, -senx * cosy, cosx * cosy) });
}

cv::Vec3f Image4DLeaf::getEulerAngles() const { return eulerAngles; }
cv::Vec3f Image4DLeaf::getPosition() const { return position; }

void Image4DLeaf::setEulerAngles(const cv::Vec3f& value)
{
    eulerAngles = value;
}

void Image4DLeaf::setPosition(const cv::Vec3f& value)
{
    position = value;
}

cv::Mat Image4DLeaf::getImage() const
{
    return image;
}

cv::Mat Image4DLeaf::getDepthMap() const
{
    return depthMap;
}

void Image4DLeaf::resizeImage()
{
    const int IMG_WIDTH = image.cols;
    const int IMG_HEIGHT = image.rows;

    DEPTH_IMG_RATIO = static_cast<float>(depthMap.cols) / IMG_WIDTH;

    if (DEPTH_IMG_RATIO == 1)
        return;

    assert(static_cast<float>(depthMap.rows) / IMG_HEIGHT == DEPTH_IMG_RATIO && "Image and cloud sizes are not proportional!");

    assert(DEPTH_IMG_RATIO < 1 && "Image is smaller than cloud!");

    cv::Size newImageSize(IMG_WIDTH * DEPTH_IMG_RATIO, IMG_HEIGHT * DEPTH_IMG_RATIO);
    cv::resize(image, image, newImageSize, cv::INTER_AREA);

    intrinsicMatrix.at<double>(0, 2) *= DEPTH_IMG_RATIO;
    intrinsicMatrix.at<double>(1, 2) *= DEPTH_IMG_RATIO;

    return;
}

bool Image4DLeaf::isLeaf() const
{
    return true;
}

void Image4DLeaf::forEachComponent(void (*func)(Image4DComponent*))
{
    func(this);
}

size_t Image4DLeaf::size() const
{
    return 1;
}

Image4DComponent* Image4DLeaf::add(Image4DComponent& item)
{
    Image4DVectorComposite* returned = new Image4DVectorComposite();
    returned->add(*this);
    returned->add(item);
    return returned;
}

Image4DComponent* Image4DLeaf::add(Image4DComponent& item, uint i)
{
    if (i == 0)
        return this->add(item);
    else
        return nullptr;
}

vector<Image4DComponent*>::iterator Image4DLeaf::begin()
{
    vector<Image4DComponent*> v;
    return v.begin();
}

vector<Image4DComponent*>::iterator Image4DLeaf::end()
{
    vector<Image4DComponent*> v;
    return v.end();
}

void Image4DLeaf::clear()
{
    Image4DLeaf();
}

Image4DComponent* Image4DLeaf::at(uint i)
{
    if (i == 0)
        return this;
    else
        return nullptr;
}

cv::Mat Image4DLeaf::getImageCovariance() const
{
    return imageCovariance;
}

void Image4DLeaf::setImageCovariance(const cv::Mat& value)
{
    imageCovariance = value;
}

cv::Mat Image4DLeaf::getDepthCovariance() const
{
    return depthCovariance;
}

void Image4DLeaf::setDepthCovariance(const cv::Mat& value)
{
    depthCovariance = value;
}
}
