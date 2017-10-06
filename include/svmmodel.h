#ifndef FACE_SVMMODEL_H
#define FACE_SVMMODEL_H

#include <opencv2/ml.hpp>

namespace face {

class SteinKernel;

struct SteinKernelParams {

    SteinKernelParams()
        : C(1)
        , gamma(1)
    {
    }
    SteinKernelParams(float C, float gamma)
        : C(C)
        , gamma(gamma)
    {
    }

    float C;
    float gamma;
    float fmeasure;
};

/**
 * @brief The SVMmodel class is a Stein kernel SVM model
 */
class SVMmodel {
public:
    SVMmodel();
    SVMmodel(const std::string& filename);

    float predict(cv::Mat& samples) const;

    bool train(const std::vector<cv::Mat>& targetPerson, const std::vector<cv::Mat>& otherPeople);

    /**
     * @brief trainAuto automatically chooses the best values for C and sigma parameters
     *        performing a grid search (without cross validation because of the small number of
     *        positive samples)
     * @param targetPerson
     * @param otherPeople
     * @param gammaGrid
     * @param CGrid
     * @return
     */
    SteinKernelParams trainAuto(const cv::Mat targetCovarMat, const std::vector<cv::Mat>& trainingSet,
        const cv::ml::ParamGrid& gammaGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA),
        const cv::ml::ParamGrid& CGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C));

    bool load(const std::string& filename);
    void save(const std::string& filename) const;

    void setC(float C);
    void setGamma(float gamma);
    void setParams(const SteinKernelParams& params);

    static cv::Mat matVectorToMat(const std::vector<cv::Mat>& data);

private:
    cv::Ptr<cv::ml::SVM> svm;

    /**
     * @brief evaluates the trained svm accuracy
     * @param validationData: Nxp CV_32FC1 matrix, where N is the number of samples
     *        and p is the number of features
     * @param groundTruth: 1xN CV_32FC1 matrix containing ground truth
     *        classification labels (1 or -1)
     * @return percentage of correct classifications (between 0 and 1)
     */
    float evaluateFMeasure(cv::Mat& validationData, const cv::Mat& groundTruth);
};

} // namespace face

#endif // SVMMODEL_H
