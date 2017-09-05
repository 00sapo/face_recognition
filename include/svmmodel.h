#ifndef SVMMODEL_H
#define SVMMODEL_H

#include <opencv2/ml.hpp>


namespace face {


class SVMmodel
{
public:
    SVMmodel();
    SVMmodel(const std::string &filePath);

    float predict(cv::InputArray samples,
                  cv::OutputArray results = cv::noArray(), int flags = 0) const;

    bool trainAuto(const cv::Ptr<cv::ml::TrainData>& data, int kFold=10,
                   cv::ml::ParamGrid gammaGrid=cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA));

private:
    cv::Ptr<cv::ml::SVM> svm;
};


}   // namespace face

#endif // SVMMODEL_H
