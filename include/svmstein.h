#ifndef FACE_SVMSTEIN_H
#define FACE_SVMSTEIN_H

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
};

/**
 * @brief The SVMmodel class is a Stein kernel SVM model
 */
class SVMStein {
public:
    SVMStein();
    SVMStein(const std::string& filename);

    /**
     * @brief predict gives the calss of an input sample
     * @param samples: row vector representing a sample
     * @return predicted class, 1 or -1
     */
    float predict(const cv::Mat &samples) const;

    /**
     * @brief getDistanceFromHyperplane gives the distance of a sample point
     *        from the optimal separating hyperplane
     * @param sample
     * @return euclidean point-plane distance
     */
    float getDistanceFromHyperplane(const cv::Mat &sample) const;

    // FIXME: change signature to match trainAuto() usage
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
    SteinKernelParams trainAuto(const cv::Mat &data, const std::vector<int> &labelsVector,
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

    cv::Mat formatDataForTraining(const std::vector<cv::Mat>& targetPerson,
        const std::vector<cv::Mat>& otherPeople) const;

    /**
     * @brief evaluates the trained svm accuracy
     * @param validationData: Nxp CV_32FC1 matrix, where N is the number of samples
     *        and p is the number of features
     * @param groundTruth: 1xN CV_32FC1 matrix containing ground truth
     *        classification labels (1 or -1)
     * @return percentage of correct classifications (between 0 and 1)
     */
    float evaluate(const cv::Mat &validationData, const cv::Mat& groundTruth);
};

} // namespace face

#endif // SVMMODEL_H
