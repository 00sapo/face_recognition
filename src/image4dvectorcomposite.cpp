#include "image4dvectorcomposite.h"
#include <image4dcomponent.h>

namespace face {
Image4DVectorComposite::Image4DVectorComposite()
{
}

bool Image4DVectorComposite::isLeaf() const
{
    return false;
}

size_t Image4DVectorComposite::size() const
{
    return vec.size();
}
void Image4DVectorComposite::forEachComponent(void (*func)(Image4DComponent*))
{
    for (Image4DComponent* image : vec)
        func(image);
}

cv::Mat Image4DVectorComposite::getImage() const
{
    return vec[0]->getImage();
}

cv::Mat Image4DVectorComposite::getDepthMap() const
{
    return vec[0]->getDepthMap();
}

size_t Image4DVectorComposite::getHeight() const
{
    return vec[0]->getHeight();
}

size_t Image4DVectorComposite::getWidth() const
{
    return vec[0]->getWidth();
}

size_t Image4DVectorComposite::getArea() const
{
    return vec[0]->getArea();
}

cv::Mat Image4DVectorComposite::get3DImage() const
{
    return vec[0]->get3DImage();
}

void Image4DVectorComposite::crop(const cv::Rect& cropRegion)
{
    for (auto& image : vec)
        image->crop(cropRegion);
}

vector<Pose> Image4DVectorComposite::getRotationMatrix() const
{
    vector<Pose> returned;
    for (auto& image : vec) {
        vector<Pose> toAppend = image->getRotationMatrix();
        returned.insert(returned.end(), toAppend.begin(), toAppend.end());
    }

    return returned;
}

cv::Vec3f Image4DVectorComposite::getEulerAngles() const
{
    return vec[0]->getEulerAngles();
}

cv::Vec3f Image4DVectorComposite::getPosition() const
{
    return vec[0]->getPosition();
}

void Image4DVectorComposite::setEulerAngles(const cv::Vec3f& value)
{
    for (auto& image : vec)
        image->setEulerAngles(value);
}

void Image4DVectorComposite::setPosition(const cv::Vec3f& value)
{
    for (auto& image : vec)
        image->setPosition(value);
}

float Image4DVectorComposite::getDepthImageRatio() const
{
    return vec[0]->getDepthImageRatio();
}

cv::Mat Image4DVectorComposite::getIntrinsicMatrix() const
{
    return vec[0]->getIntrinsicMatrix();
}

std::string Image4DVectorComposite::getName()
{
    return vec[0]->getName();
}

void Image4DVectorComposite::setName(std::string name)
{
    return vec[0]->setName(name);
}

void Image4DVectorComposite::depthForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI)
{
    for (auto& image : vec)
        image->depthForEach(function, ROI);
}

void Image4DVectorComposite::imageForEach(const std::function<void(int, int, boost::any&)>& function, const cv::Rect& ROI)
{
    for (auto& image : vec)
        image->imageForEach(function, ROI);
}

cv::Mat Image4DVectorComposite::getDepthCovariance() const
{
    return depthCovariance;
}

void Image4DVectorComposite::setDepthCovariance(const cv::Mat& value)
{
    depthCovariance = value;
}

cv::Mat Image4DVectorComposite::getImageCovariance() const
{
    return imageCovariance;
}

void Image4DVectorComposite::setImageCovariance(const cv::Mat& value)
{
    imageCovariance = value;
}

vector<Image4DComponent*> Image4DVectorComposite::getVec() const
{
    return vec;
}

void Image4DVectorComposite::setVec(const vector<Image4DComponent*>& value)
{
    vec = value;
}

void Image4DVectorComposite::resizeImage()
{
    for (auto& image : vec)
        image->resizeImage();
}

Image4DComponent* Image4DVectorComposite::add(Image4DComponent& item)
{
    vec.push_back(&item);
    return this;
}

Image4DComponent* Image4DVectorComposite::add(Image4DComponent& item, uint i)
{
    vec.at(i)->add(item);
    return this;
}

vector<Image4DComponent*>::iterator Image4DVectorComposite::begin()
{
    return vec.begin();
}

vector<Image4DComponent*>::iterator Image4DVectorComposite::end()
{
    return vec.end();
}

void Image4DVectorComposite::clear()
{
    vec = vector<Image4DComponent*>();
}

Image4DComponent* face::Image4DVectorComposite::at(uint i)
{
    return vec.at(i);
}

void face::Image4DVectorComposite::setDepthMap(const cv::Mat& value)
{
    vec.at(0)->setDepthMap(value);
}

void face::Image4DVectorComposite::setImage(const cv::Mat& value)
{
    vec.at(0)->setImage(value);
}
}
