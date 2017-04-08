#include "backgroundsegmentation.h"

BackgroundSegmentation::BackgroundSegmentation()
{
}

float BackgroundSegmentation::findTreshold()
{
}

float BackgroundSegmentation::getTreshold() const
{
    return treshold;
}

void BackgroundSegmentation::setTreshold(float value)
{
    treshold = value;
}

std::vector<cv::Mat> BackgroundSegmentation::getCollectionRGB() const
{
    return collectionRGB;
}

void BackgroundSegmentation::setCollectionRGB(const std::vector<cv::Mat>& value)
{
    collectionRGB = value;
}

void BackgroundSegmentation::setCollectionDepth(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& value)
{
    collectionDepth = value;
}

std::vector<pcl::PointCloud::Ptr> BackgroundSegmentation::getCollectionDepth() const
{
    return collectionDepth;
}
