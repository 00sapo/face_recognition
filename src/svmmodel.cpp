#include "svmmodel.h"

namespace ml = cv::ml;
using std::vector;
using cv::Mat;

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



namespace face {


/**
 * @brief The SteinKernel class is a custom SVM kernel for
 *        Lie group of Riemannian manifold, the space of symmetric
 *        positive definite matrixes
 */
class SteinKernel : public ml::SVM::Kernel {

public:
    SteinKernel(ml::SvmParams params = ml::SvmParams()) : params(params) { }

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
            results[i] = std::exp(-params.gamma * s);
        }
    }


    int getType() const
    {
        return ml::SVM::CUSTOM;
    }

    ml::SvmParams params;
};


SVMmodel::SVMmodel()
{
    svm = ml::SVM::create();
    svm->setCustomKernel(cv::Ptr<SteinKernel>());
}

SVMmodel::SVMmodel(const std::string &filename)
{
    if(!load(filename))
        std::cout << "Error. Failed loading pretrained SVM model." << std::endl;
}

float SVMmodel::predict(cv::InputArray samples,
                        cv::OutputArray results,
                        int flags) const
{
    return svm->predict(samples, results, flags);
}

bool SVMmodel::trainAuto(const vector<Mat> &targetPerson, const vector<Mat> &otherPeople, int kFold,
                         ml::ParamGrid gammaGrid)
{
    // conversion from vector<Mat> to Mat....
    const auto personSize = targetPerson.size();
    const auto trainDataHeight = personSize + otherPeople.size();
    const auto trainDataWidth  = targetPerson[0].rows * targetPerson[0].cols;

    Mat data(trainDataHeight, trainDataWidth, CV_32F);


    Mat labels(trainDataHeight, 1, CV_32F);

    // fill data rows with targetPerson
    for (auto i = 0; i < personSize; ++i) {
        auto iter = targetPerson[i].begin<float>();
        for (auto j = 0; j < trainDataWidth; ++j, ++iter) {
            data.at<float>(i,j) = *iter;
        }
        labels.at<float>(i,0) = 1.0f;
    }

    // fill data rows with otherPeople
    for (auto i = personSize; i < trainDataHeight; ++i) {
        auto iter = otherPeople[i].begin<float>();
        for (auto j = 0; j < trainDataWidth; ++j, ++iter) {
            data.at<float>(i,j) = *iter;
        }
        labels.at<float>(i,0) = -1.0f;
    }

    auto trainData = ml::TrainData::create(data, ml::ROW_SAMPLE, labels);
    return svm->trainAuto(trainData, kFold, ml::SVM::getDefaultGrid(ml::SVM::C), gammaGrid);
}

bool SVMmodel::load(const std::string &filename)
{
    svm = ml::SVM::load(filename);
    return svm != nullptr;
}

void SVMmodel::save(const std::string &filename)
{
    svm->save(filename);
}

}   // namespace face
