#ifndef SVMMODEL_H
#define SVMMODEL_H

#include <opencv2/ml.hpp>


namespace face {


class SVMmodel
{
public:
    SVMmodel();
    SVMmodel(const std::string &filename);

    float predict(cv::InputArray &samples) const;

    bool train    (const std::vector<cv::Mat> &targetPerson, const std::vector<cv::Mat> &otherPeople);

    bool trainAuto(const std::vector<cv::Mat> &targetPerson, const std::vector<cv::Mat> &otherPeople,
                   const cv::ml::ParamGrid &gammaGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA),
                   const cv::ml::ParamGrid &CGrid     = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C));

    bool load(const std::string &filename);
    void save(const std::string &filename) const;

private:
    cv::Ptr<cv::ml::SVM> svm;

    cv::Ptr<cv::ml::TrainData> formatDataForTraining(const std::vector<cv::Mat> &targetPerson,
                                                     const std::vector<cv::Mat> &otherPeople) const;

    float evaluate(cv::InputArray &validationData, const std::vector<int> &groundTruth);
};


}   // namespace face

#endif // SVMMODEL_H
