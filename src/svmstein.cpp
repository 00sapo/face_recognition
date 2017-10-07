#include "svmstein.h"
#include "numeric"

namespace ml = cv::ml;

using cv::Mat;
using std::cout;
using std::endl;
using std::vector;

namespace face {

/**
 * @brief The SteinKernel class is a custom SVM kernel for
 *        Lie group of Riemannian manifold, the space of symmetric
 *        positive definite matrixes
 */
class SteinKernel : public ml::SVM::Kernel {

public:
    SteinKernel(float gamma)
        : gamma(gamma)
    {
    }

    /**
     * @brief calc computes the Stein kernel function.
     *        results[i] = <h(x),h(samples[i])>, for i = 0,...,N-1
     *        where <X,Y> denotes the inner product.
     * @param N number of input samples (matrixes in our case)
     * @param numOfFeatures length of sample (# of matrixes elements in our case)
     * @param samples input samples
     * @param x another input sample
     * @param results array of the results
     */
    void calc(int N, int numOfFeatures, const float* samples, const float* x, float* results)
    {
        Mat Y(16, 16, CV_32F, (void*)(x));

        for (int i = 0; i < N; i++) {
            Mat X(16, 16, CV_32F, (void*)(samples + numOfFeatures * i));
            auto A = X + Y;
            double S = std::log10(cv::determinant(A / 2)) - std::log10(cv::determinant(X * Y)) / 2;
            results[i] = std::exp(-gamma * S);
        }
    }

    int getType() const
    {
        return ml::SVM::CUSTOM;
    }

    float gamma;
};

SVMStein::SVMStein()
{
    svm = ml::SVM::create();

    svm->setCustomKernel(new SteinKernel(1));
    svm->setType(ml::SVM::C_SVC);
    svm->setC(1);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

SVMStein::SVMStein(const std::string& filename)
{
    if (!load(filename))
        std::cout << "Error. Failed loading pretrained SVM model." << std::endl;
}

float SVMStein::predict(const Mat &samples) const
{
    Mat res;
    svm->predict(samples, res);
    return res.at<float>(0);
}


float SVMStein::getDistanceFromHyperplane(const cv::Mat &sample) const
{
    Mat res;
    svm->predict(sample, res, ml::StatModel::RAW_OUTPUT);
    return res.at<float>(0);
}

bool SVMStein::train(const Mat &data, const vector<int> &labelsVector)
{
    auto trainData = ml::TrainData::create(data, ml::ROW_SAMPLE, Mat(labelsVector, true));
    return svm->train(trainData);
}

SteinKernelParams SVMStein::trainAuto(const Mat &data, const vector<int> &labelsVector,
                                      const ml::ParamGrid& gammaGrid, const ml::ParamGrid& CGrid)
{
    Mat labels(labelsVector, true);

    auto trainData = ml::TrainData::create(data, ml::ROW_SAMPLE, labels);

    vector<double> bestGamma, bestC;
    float bestScore = 0;
    for (auto gamma = gammaGrid.minVal; gamma < gammaGrid.maxVal; gamma *= gammaGrid.logStep) {
        setGamma(gamma);
        for (auto C = CGrid.minVal; C < CGrid.maxVal; C *= CGrid.logStep) {
            std::cout << "C = " << C << "; gamma = " << gamma << std::endl;
            setC(C);
            cout << "Training..." << endl;
            svm->train(trainData);
            cout << "Done!" << endl;
            auto score = evaluate(data, labels);
            cout << "Score: " << score << endl;
            if (bestScore < score) {
                bestGamma = { gamma };
                bestC = { C };
                bestScore = score;
            } else if (bestScore == score) {
                bestC.push_back(C);
                bestGamma.push_back(gamma);
            }
        }
    }
    double sum = std::accumulate(bestC.begin(), bestC.end(), 0.0);
    double C = sum / bestC.size();
    sum = std::accumulate(bestGamma.begin(), bestGamma.end(), 0.0);
    double gamma = sum / bestGamma.size();

    setC(C);
    setGamma(gamma);
    svm->train(trainData);
    auto score = evaluate(data, labels);

    std::cout << "score obtained by avaraging best parameters: " << score << std::endl;
    return SteinKernelParams(C, gamma);
}

bool SVMStein::load(const std::string &filename)
{
    svm = ml::SVM::load(filename);
    return svm != nullptr;
}

void SVMStein::save(const std::string &filename) const
{
    svm->save(filename);
}

void SVMStein::setC(float C)
{
    svm->setC(C);
}

void SVMStein::setGamma(float gamma)
{
    svm->setCustomKernel(new SteinKernel(gamma));
}

void SVMStein::setParams(const SteinKernelParams& params)
{
    setC(params.C);
    setGamma(params.gamma);
}


float SVMStein::evaluate(const Mat &validationData, const Mat &groundTruth)
{
    assert(validationData.rows == groundTruth.rows && "Attention! Validation and ground truth arrays should be of the same size");

    const int N = validationData.rows;
    int correctClassification = 0;
    for (int i = 0; i < N; ++i) {
        const auto row = validationData.row(i);
        auto prediction = predict(row);
        auto truth = float(groundTruth.at<int>(i, 0));
        if (prediction == truth)
            ++correctClassification;
    }

    return correctClassification / float(N);
}


} // namespace face
