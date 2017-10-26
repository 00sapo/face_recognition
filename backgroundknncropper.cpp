#include "backgroundknncropper.h"

namespace face {
BackgroundKNNCropper::BackgroundKNNCropper()
{
}

void BackgroundKNNCropper::cropImage(cv::Mat& maskKNN, cv::Rect& roi)
{
    int x = 0, y = 0, width = 0, height = 0;
    int NONZERO_TH = 5;
    for (int i = 0; i < maskKNN.rows; ++i) {
        // looking for first non empty row

        int nonzero = 0;
        for (int j = 0; j < maskKNN.cols; ++j) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                nonzero++;
            }
        }
        if (nonzero > NONZERO_TH) {
            y = i;
            break;
        }
    }

    for (int j = 0; j < maskKNN.cols; ++j) {
        // looking for first non empty col
        int nonzero = 0;
        for (int i = 0; i < maskKNN.rows; ++i) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                nonzero++;
            }
        }
        if (nonzero > NONZERO_TH) {
            x = j;
            break;
        }
    }

    for (int i = maskKNN.rows - 1; i > 0; --i) {
        // looking for last non empty row

        int nonzero = 0;
        for (int j = 0; j < maskKNN.cols; ++j) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                nonzero++;
            }
        }
        if (nonzero > NONZERO_TH) {
            height = i - y;
            break;
        }
    }

    for (int j = maskKNN.cols - 1; j > 0; --j) {
        // looking for last non empty col

        int nonzero = 0;
        for (int i = 0; i < maskKNN.rows; ++i) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                nonzero++;
            }
        }
        if (nonzero > NONZERO_TH) {
            width = j - x;
            break;
        }
    }
    width = (width == 0 ? maskKNN.cols - x : width);
    height = (height == 0 ? maskKNN.rows - y : height);
    roi = cv::Rect(x, y, width, height);
}

bool face::BackgroundKNNCropper::filter()
{
    cv::Mat frame, maskKNN;
    for (Image4DComponent* id : *imageSet) {
        subtractor = cv::createBackgroundSubtractorKNN(1, 400, true);
        // creating model
        for (Image4DComponent* img : *id) {
            cv::blur(img->getImage(), frame, cv::Size(9, 9));
            subtractor->apply(frame, maskKNN);
        }

        // using it
        for (Image4DComponent* img : *id) {
            cv::blur(img->getImage(), frame, cv::Size(9, 9));
            subtractor->apply(frame, maskKNN);

            cv::Rect roi;
            cropImage(maskKNN, roi);

            cv::Mat filtered = img->getImage()(roi);
            img->setImage(filtered);

            filtered = img->getDepthMap()(roi);
            img->setDepthMap(filtered);
        }
    }
    return true;
}

Image4DComponent* face::BackgroundKNNCropper::getImage4DComponent() const
{
    return imageSet;
}

void face::BackgroundKNNCropper::setImage4DComponent(Image4DComponent* value)
{
    imageSet = value;
}
}