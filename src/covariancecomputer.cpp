#include "covariancecomputer.h"
#include <lbp.h>

using cv::Mat;
using std::vector;

namespace face {

using MatSet = vector<vector<Mat>>;

CovarianceComputer::CovarianceComputer()
{
}

bool CovarianceComputer::filter()
{
    bool result = false;

    Image4DComponent* backupImage = imageSet;
    if ((imageSet->isLeaf() && leafCovarianceComputer)
        || imageSet->at(0)->isLeaf()) {
        // leaf or second-last level
        result = setToNormalizedCovariance(*imageSet);
        if (!result)
            return false;
    } else {
        for (Image4DComponent* image : *backupImage) {
            imageSet = image;
            filter();
        }
    }
    imageSet = backupImage;

    return result;
}

Image4DComponent* CovarianceComputer::getImage4DComponent() const
{
    return imageSet;
}

void CovarianceComputer::setImage4DComponent(Image4DComponent* value)
{
    imageSet = value;
}

bool CovarianceComputer::isLeafCovarianceComputer()
{
    return leafCovarianceComputer;
}

void CovarianceComputer::setLeafCovarianceComputer(bool value)
{
    leafCovarianceComputer = value;
}

bool CovarianceComputer::setToNormalizedCovariance(Image4DComponent& set)
{
    cv::Mat imageCovariance, depthCovariance;

    const int SET_SIZE = set.size();
    if (SET_SIZE == 0) {
        imageCovariance = Mat::zeros(16, 16, CV_32FC1);
        depthCovariance = Mat::zeros(16, 16, CV_32FC1);
        return false;
    }

    MatSet imageBlocks(16);
    MatSet depthBlocks(16);
    for (int i = 0; i < 16; ++i) {
        imageBlocks[i].resize(SET_SIZE);
        depthBlocks[i].resize(SET_SIZE);
    }

    Mat imageMean(16, SET_SIZE, CV_32FC1);
    Mat depthMean(16, SET_SIZE, CV_32FC1);

    // for each face in the set...
    for (uint i = 0; i < set.size(); i++) {
        Image4DComponent* image4d = set.at(i);

        assert(!image4d->getImage().empty() && !image4d->getDepthMap().empty() && "ERROR! Empty image!!");

        // compute 4x4 blocks size
        const auto HEIGHT = image4d->getHeight();
        const auto WIDTH = image4d->getWidth();
        const auto BLOCK_H = HEIGHT / 4;
        const auto BLOCK_W = WIDTH / 4;

        // for each of the 16 blocks of the face...
        for (size_t y = 0, q = 0; q < 4; y += BLOCK_H, ++q) {
            for (size_t x = 0, p = 0; p < 4; x += BLOCK_W, ++p) {

                // crop block region
                cv::Rect roi(x, y, BLOCK_W, BLOCK_H);
                Mat image = image4d->getImage()(roi);
                Mat depth = image4d->getDepthMap()(roi);

                // compute LBP of the block
                auto imageHist = OLBPHist(image);
                auto depthHist = OLBPHist(depth);

                imageBlocks[p + 4 * q][i] = imageHist;
                depthBlocks[p + 4 * q][i] = depthHist;

                imageMean.at<float>(p + 4 * q, i) = mean(imageHist)[0];
                depthMean.at<float>(p + 4 * q, i) = mean(depthHist)[0];
            }
        }
    }

    imageCovariance = Mat(16, 16, CV_32FC1);
    depthCovariance = Mat(16, 16, CV_32FC1);

    // Computing covariances.
    // OpenCV cv::calcCovarMatrix() could be used but it's very hard to obtain the same result
    // because data representation should be changed requiring extra coding and runtime work
    for (int p = 0; p < 16; ++p) {
        for (int q = 0; q < 16; ++q) {
            float imageValue = 0, depthValue = 0;
            for (int i = 0; i < SET_SIZE; ++i) {
                imageValue += (imageBlocks[p][i] - imageMean.at<float>(p, i)).dot(imageBlocks[q][i] - imageMean.at<float>(q, i));
                depthValue += (depthBlocks[p][i] - depthMean.at<float>(p, i)).dot(depthBlocks[q][i] - depthMean.at<float>(q, i));
            }
            imageCovariance.at<float>(p, q) = imageValue / SET_SIZE;
            depthCovariance.at<float>(p, q) = depthValue / SET_SIZE;
        }
    }
    Mat normalizedDepth, normalizedImage;
    normalize(depthCovariance, normalizedDepth);
    set.setDepthCovariance(normalizedDepth);
    normalize(imageCovariance, normalizedImage);
    set.setImageCovariance(normalizedImage);

    return true;
}
}
