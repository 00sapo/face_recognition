#include "svmstein.h"
#include "numeric"

namespace ml = cv::ml;

using cv::Mat;
using std::string;
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



/**----------------------------------------------------------
 * ------------------------ SVMStein ------------------------
 * ----------------------------------------------------------*/

SVMStein::SVMStein()
{
    svm = ml::SVM::create();

    svm->setCustomKernel(new SteinKernel(1));
    svm->setType(ml::SVM::C_SVC);
    svm->setC(1);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

SVMStein::SVMStein(const string& filename)
{
    if (!load(filename))
        std::cout << "Error. Failed loading pretrained SVM model." << std::endl;
}

float SVMStein::predict(const Mat& samples) const
{
    return svm->predict(samples);
}

float SVMStein::getDistanceFromHyperplane(const Mat& sample) const
{
    Mat res;
    svm->predict(sample, res, ml::StatModel::RAW_OUTPUT);
    return res.at<float>(0);
}

bool SVMStein::train(const Mat& data, const vector<int>& labelsVector)
{
    auto trainData = ml::TrainData::create(data, ml::ROW_SAMPLE, Mat(labelsVector, true));
    return svm->train(trainData);
}

SteinKernelParams SVMStein::trainAuto(const Mat& dataTr, const vector<int>& labelsVector, const Mat& dataVal,
                                      const vector<int>& groundTruth, const ml::ParamGrid& gammaGrid, const ml::ParamGrid& CGrid)
{
    auto trainData = ml::TrainData::create(dataTr, ml::ROW_SAMPLE, Mat(labelsVector, true));

    vector<double> bestGamma, bestC;
    float bestScore = 0;
    for (auto gamma = gammaGrid.minVal; gamma < gammaGrid.maxVal; gamma *= gammaGrid.logStep) {
        setGamma(gamma);
        for (auto C = CGrid.minVal; C < CGrid.maxVal; C *= CGrid.logStep) {
            //std::cout << "C = " << C << "; gamma = " << gamma << std::endl;
            setC(C);
            //std::cout << "Training..." << std::endl;
            svm->train(trainData);
            float fscore = evaluateFMeasure(dataVal, groundTruth);
            //std::cout << "Score: " << fscore << std::endl;
            if (bestScore < fscore) {
                bestGamma = { gamma };
                bestC = { C };
                bestScore = fscore;
            } else if (bestScore == fscore) {
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
    auto fscore = evaluateFMeasure(dataVal, labelsVector);

    std::cout << "Best score: " << bestScore << std::endl;
    std::cout << "Best C: " << C << "\nBest gamma: " << gamma << std::endl;
    std::cout << "score obtained by avaraging best parameters: " << fscore << std::endl;
    return SteinKernelParams(C, gamma);
}

bool SVMStein::load(const string& filename)
{
    svm = ml::SVM::load(filename);
    svm->setType(ml::SVM::CUSTOM);
    return svm != nullptr;
}

void SVMStein::save(const string& filename) const
{
    svm->setType(ml::SVM::POLY);
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

float SVMStein::evaluateFMeasure(const Mat& dataVal, const vector<int>& groundTruth)
{
    /*
   * 1) Recall:       true positives (0 or 1 in this case) divided by all real positives
   * (true positives + false negatives, exactly 1 in this case).
   * 2) Precision:    true positives divided by all detected positives (true positives + false positives).
   * 3) F-measure:    harmonic mean between precision and recall, so 2*1/(1/p+1/r)
   *
   * Note: in this case recall can be only 0 or 1 and, in general,
   * if recall is 0 then precision is 0, if recall is 1 precision is 1/(1+false positives).
   * So in this case F-measure will be:
   * A) true positives = 1 => (2*false positives+2)/(false positives + 2)
   * B) true positives = 0 => 0
   *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   *|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   * NON È VERO SE SI USANO I CLUSTER DELLE POSE IN QUESTO MODO...
   */

    int truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;
    for (auto i = 0; i < dataVal.rows; ++i) {
        float prediction = predict(dataVal.row(i));
        if (prediction == 1) {
            if (prediction == groundTruth[i])
                ++truePositives;
            else
                ++falsePositives;
        } else if (prediction == -1) {
            if (prediction == groundTruth[i])
                ++trueNegatives;
            else
                ++falseNegatives;
        } else {
            std::cerr << "Warning! SVM prediction != 1 and != -1" << std::endl;
        }

    }

    return 2 * truePositives / (float)(truePositives + falseNegatives + falsePositives);
}

} // namespace face
