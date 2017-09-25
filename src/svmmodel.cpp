#include "svmmodel.h"

namespace ml = cv::ml;
using std::vector;
using cv::Mat;
/*
namespace cv {
namespace ml {

struct SvmParams
{
    int         svmType;
    int         kernelType;
    double      gamma;
    double      coef0;
    double      degree;
    double      C;
    double      nu;
    double      p;
    Mat         classWeights;
    TermCriteria termCrit;

    SvmParams();

    SvmParams( int _svmType, int _kernelType,
            double _degree, double _gamma, double _coef0,
            double _Con, double _nu, double _p,
            const Mat& _classWeights, TermCriteria _termCrit );
};

}
}
*/


namespace face {


/**
 * @brief The SteinKernel class is a custom SVM kernel for
 *        Lie group of Riemannian manifold, the space of symmetric
 *        positive definite matrixes
 */
class SteinKernel : public ml::SVM::Kernel {

public:
    SteinKernel(float gamma) : gamma(gamma) { }

    /**
     * @brief cv::ml::SVM::Kernel::calc compute the Stein kernel function.
     *        It uses only gamma parameter
     * @param vcount number of samples (matrixes in our case)
     * @param n length of sample (# of matrixes elements in our case)
     * @param x one of the input sample
     * @param y another input sample
     * @param results array of the results
     */
    void calc(int vcount, int n, const float* x, const float* y, float* results)
    {
       Mat Y(16, 16, CV_32F, (void*)y);

        for (int i = 0; i < vcount; i++) {
            Mat X(16, 16, CV_32F, (void*)(x + n * i));
            auto A = X + Y;
            double   s = std::log10(determinant(A * 0.5)) - 0.5 * std::log10(determinant(X * Y));
            results[i] = std::exp(-gamma * s);
        }
    }


    int getType() const
    {
        return ml::SVM::CUSTOM;
    }

    float gamma;
};


SVMmodel::SVMmodel()
{
    svm = ml::SVM::create();
    svm->setCustomKernel(new SteinKernel(1));
    svm->setType(ml::SVM::C_SVC);
    svm->setC(1);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

}

SVMmodel::SVMmodel(const std::string &filename)
{
    if(!load(filename))
        std::cout << "Error. Failed loading pretrained SVM model." << std::endl;
}

float SVMmodel::predict(cv::InputArray &samples) const
{
    Mat res;
    svm->predict(samples, res);
    return res.at<float>(0);
}


bool SVMmodel::train(const std::vector<cv::Mat> &targetPerson, const vector<Mat> &otherPeople)
{
    auto trainData = formatDataForTraining(targetPerson, otherPeople);
    return svm->train(trainData);
}

bool SVMmodel::trainAuto(const vector<Mat> &targetPerson, const vector<Mat> &otherPeople,
                         const ml::ParamGrid &gammaGrid, const ml::ParamGrid &CGrid)
{
    auto trainData = formatDataForTraining(targetPerson, otherPeople);

    double bestGamma, bestC;
    float bestScore = 0;
    for (auto gamma = gammaGrid.minVal; gamma < gammaGrid.maxVal; gamma *= gammaGrid.logStep) {
        svm->setCustomKernel(new SteinKernel(gamma));
        for (auto C = CGrid.minVal; C < CGrid.maxVal; C *= CGrid.logStep) {
           svm->setC(C);
           svm->train(trainData);
           if (bestScore < evaluate(targetPerson,vector<int>())) {
               bestGamma = gamma;
               bestC = C;
           }
        }
    }
}

bool SVMmodel::load(const std::string &filename)
{
    svm = ml::SVM::load(filename);
    return svm != nullptr;
}

void SVMmodel::save(const std::string &filename) const
{
    svm->save(filename);
}

cv::Ptr<ml::TrainData> SVMmodel::formatDataForTraining(const vector<Mat> &targetPerson,
                                                       const vector<Mat> &otherPeople) const
{
    std::cout << "conversion from vector<Mat> to Mat..." << std::endl;
    const auto personSize = targetPerson.size();
    const auto samples  = personSize + otherPeople.size();
    const auto features = targetPerson[0].rows * targetPerson[0].cols;

    Mat data(samples, features, CV_32FC1);

    vector<int> lab;

    std::cout << "  fill data rows with targetPerson" << std::endl;
    for (auto i = 0; i < personSize; ++i) {
        auto iter = targetPerson[i].begin<float>();
        for (auto j = 0; j < features; ++j, ++iter) {
            data.at<float>(i,j) = *iter;
        }
        lab.push_back(1);
    }

    std::cout << "  fill data rows with otherPeople" << std::endl;
    for (auto i = personSize; i < samples; ++i) {
        auto iter = otherPeople[i-personSize].begin<float>();
        for (auto j = 0; j < features; ++j, ++iter) {
            data.at<float>(i,j) = *iter;
        }
        lab.push_back(-1);
    }

    Mat labels(lab,true);

    return ml::TrainData::create(data, ml::ROW_SAMPLE, labels);
}

float SVMmodel::evaluate(cv::InputArray &validationData, const std::vector<int> &groundTruth)
{
    return 0;
}


}   // namespace face
