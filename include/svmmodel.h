#ifndef SVMMODEL_H
#define SVMMODEL_H

#include <opencv2/ml.hpp>


namespace face {


class SVMmodel
{
public:
    SVMmodel();
    SVMmodel(const std::string &filename);

    float predict(cv::InputArray samples,
                  cv::OutputArray results = cv::noArray(), int flags = 0) const;

    bool train    (const std::vector<cv::Mat> targetPerson, const std::vector<cv::Mat> &otherPeople);

    bool trainAuto(const std::vector<cv::Mat> &targetPerson, const std::vector<cv::Mat> &otherPeople,
                   int kFold, cv::ml::ParamGrid gammaGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA));

    bool load(const std::string &filename);
    void save(const std::string &filename);

private:
    cv::Ptr<cv::ml::SVM> svm;
};


}   // namespace face

#endif // SVMMODEL_H
