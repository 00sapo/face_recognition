#include "backgroundsegmentation.h"
#include "pcl/ml/kmeans.h"

BackgroundSegmentation::BackgroundSegmentation()
{
}

float BackgroundSegmentation::findTreshold()
{
    pcl::Kmeans classifier(imageDepth->size(), 1);

    for (int i = 0; i < imageDepth->size(); ++i) {
        pcl::Kmeans::Point p = { imageDepth->at(i).z };
        classifier.addDataPoint(p);
    }

    classifier.kMeans();

    /* TODO verificare se bisogna prendere l'indice 0 o 1*/
    pcl::Kmeans::SetPoints background = classifier.clusters_to_points_[0];

    float minimum = 0xffff;
    for (auto& t : background) {
        if (imageDepth->at(t).z < minimum)
            thresold = imageDepth->at(t).z;
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

void BackgroundSegmentation::setImageDepth(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& value)
{
    imageDepth = value;
}

pcl::PointCloud::Ptr BackgroundSegmentation::getImageDepth() const
{
    return imageDepth;
}
