#include "backgroundsegmentation.h"
#include "pcl/ml/kmeans.h"

BackgroundSegmentation::BackgroundSegmentation()
    : Kmeans(0, 1)
{
}

float BackgroundSegmentation::findTreshold()
{

    for (unsigned int i = 0; i < imageDepth->size(); ++i) {
        pcl::Kmeans::Point p = { imageDepth->at(i).z };
        this->addDataPoint(p);
    }

    this->kMeans();

    /* TODO verificare se bisogna prendere l'indice 0 o 1*/
    pcl::Kmeans::SetPoints background = this->clusters_to_points_[0];

    float minimum = 0xffff;
    for (auto& t : background) {
        if (imageDepth->at(t).z < minimum)
            minimum = imageDepth->at(t).z;
    }

    this->threshold = minimum;

    return this->threshold;
}

float BackgroundSegmentation::getTreshold() const
{
    return threshold;
}

void BackgroundSegmentation::setTreshold(float value)
{
    threshold = value;
}

cv::Mat BackgroundSegmentation::getImageRGB() const
{
    return imageRGB;
}

void BackgroundSegmentation::setImageRGB(cv::Mat& value)
{
    imageRGB = value;
}

void BackgroundSegmentation::setImageDepth(const pcl::PointCloud<pcl::PointXYZ>::Ptr& value)
{
    imageDepth = value;
    num_points_ = imageDepth->size();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BackgroundSegmentation::getImageDepth() const
{
    return imageDepth;
}
