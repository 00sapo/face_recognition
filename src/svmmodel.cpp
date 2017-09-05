#include "svmmodel.h"

namespace ml = cv::ml;

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
       cv::Mat Y(16, 16, CV_32F, (void*)y);

        for (int i = 0; i < vcount; i++) {
            cv::Mat X(16, 16, CV_32F, (void*)(x + n * i));
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

SVMmodel::SVMmodel(const std::string &filePath)
{
    svm = ml::SVM::load(filePath);
}

float SVMmodel::predict(cv::InputArray samples,
                        cv::OutputArray results,
                        int flags) const
{
    return 0;
}

bool SVMmodel::trainAuto(const cv::Ptr<ml::TrainData>& data, int kFold,
                         ml::ParamGrid gammaGrid)
{
    return svm->trainAuto(data, kfold, ml::SVM::getDefaultGrid(ml::SVM::C), gammaGrid);
}

}   // namespace face
