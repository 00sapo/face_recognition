#include "backgroundknncropper.h"

namespace face {
BackgroundKNNCropper::BackgroundKNNCropper()
{
}

void BackgroundKNNCropper::cropImage(cv::Mat& maskKNN, cv::Rect& roi)
{
    int x = 0, y = 0, width = 0, height = 0;
    bool flag = false;
    for (int i = 0; i < maskKNN.rows; ++i) {
        // looking for first non empty row

        for (int j = 0; j < maskKNN.cols; ++j) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                flag = true;
                y = i;
                break;
            }
        }
        if (flag)
            break;
    }

    flag = false;
    for (int j = 0; j < maskKNN.cols; ++j) {
        // looking for first non empty col

        for (int i = 0; i < maskKNN.rows; ++i) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                flag = true;
                x = j;
                break;
            }
        }
        if (flag)
            break;
    }

    flag = false;
    for (int i = maskKNN.rows - 1; i > 0; --i) {
        // looking for last non empty row

        for (int j = 0; j < maskKNN.cols; ++j) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                flag = true;
                height = i - y;
                break;
            }
        }
        if (flag)
            break;
    }

    flag = false;
    for (int j = maskKNN.cols - 1; j > 0; --j) {
        // looking for last non empty col

        for (int i = 0; i < maskKNN.rows; ++i) {
            if (maskKNN.at<uchar>(i, j) == 255) {
                flag = true;
                width = j - x;
                break;
            }
        }
        if (flag)
            break;
    }
    width = (width == 0 ? maskKNN.cols - x : width);
    height = (height == 0 ? maskKNN.rows - y : height);
    roi = cv::Rect(x, y, width, height);
}

bool face::BackgroundKNNCropper::filter()
{
    std::cout << "Debugging: removing background using KNN" << std::endl;
    cv::Mat frame, maskKNN;
    for (Image4DComponent* id : *imageSet) {
        std::cout << "processing " << id->getName() << std::endl;
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
