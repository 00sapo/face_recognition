#include "svmmodel.h"
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

SVMmodel::SVMmodel()
{
    svm = ml::SVM::create();

    svm->setCustomKernel(new SteinKernel(1));
    svm->setType(ml::SVM::C_SVC);
    svm->setC(1);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

SVMmodel::SVMmodel(const std::string& filename)
{
    if (!load(filename))
        std::cout << "Error. Failed loading pretrained SVM model." << std::endl;
}

float SVMmodel::predict(Mat& samples) const
{
    return svm->predict(samples);
}

bool SVMmodel::train(const std::vector<cv::Mat>& trainingSet, int targetIndex)
{
    Mat trainMatrix = matVectorToMat(trainingSet);
    vector<int> labelsVector(trainingSet.size(), -1);
    labelsVector.at(targetIndex) = 1;

    //    auto trainData = ml::TrainData::create(trainMatrix, ml::ROW_SAMPLE, Mat(labelsVector, true));
    return svm->train(trainMatrix, ml::ROW_SAMPLE, labelsVector);
}

SteinKernelParams SVMmodel::trainAuto(const vector<Mat>& trainingSet, const int targetIndex, Mat targetDepthCovar,
    const ml::ParamGrid& gammaGrid, const ml::ParamGrid& CGrid)
{
    /*
    cout << "Training with leave-one-out targetIndex=" << targetIndex << endl;

    vector<double> bestGamma, bestC;
    float bestScore = 0;
    for (auto gamma = gammaGrid.minVal; gamma < gammaGrid.maxVal; gamma *= gammaGrid.logStep) {
        setGamma(gamma);
        for (auto C = CGrid.minVal; C < CGrid.maxVal; C *= CGrid.logStep) {
            std::cout << "C = " << C << "; gamma = " << gamma << std::endl;
            setC(C);
            train(trainingSet, targetIndex);
            float fscore = evaluateFMeasure(targetDepthCovar, targetIndex);
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
    auto score = evaluateFMeasure(validationData,
        labels(cv::Rect(0, targetPerson.size(), 1, otherPeople.size())));

    std::cout << "score obtained by avaraging best parameters: " << score << std::endl;
    return SteinKernelParams(C, gamma);
    */
}

bool SVMmodel::load(const std::string& filename)
{
    svm = ml::SVM::load(filename);
    return svm != nullptr;
}

void SVMmodel::save(const std::string& filename) const
{
    svm->save(filename);
}

void SVMmodel::setC(float C)
{
    svm->setC(C);
}

void SVMmodel::setGamma(float gamma)
{
    svm->setCustomKernel(new SteinKernel(gamma));
}

void SVMmodel::setParams(const SteinKernelParams& params)
{
    setC(params.C);
    setGamma(params.gamma);
}

float SVMmodel::evaluateFMeasure(Mat& targetDepthCovar, const int targetIndex)
{

    /* 1) Recall:       true positives (0 or 1 in this case) divided by all real positives
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
   *
   *
    targetDepthCovar.reshape(targetDepthCovar.rows * targetDepthCovar.cols, 0);
    float prediction = predict(targetDepthCovar);
    int truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;
    if (prediction == 1) {
        if (prediction == truth)
            ++truePositives;
        else
            ++falsePositives;
    } else if (prediction == -1) {
        if (prediction == truth)
            ++trueNegatives;
        else
            ++falseNegatives;
    }
    float fmeasure = 2 * truePositives / (float)(truePositives + falseNegatives + falsePositives);
    */
}

Mat SVMmodel::matVectorToMat(const vector<Mat>& data)
{
    const int height = data.size();
    const int width = data[0].rows * data[0].cols;
    Mat matrix(height, width, data[0].type());
    for (auto i = 0; i < height; ++i) {
        auto iter = data[i].begin<float>();
        for (auto j = 0; j < width; ++j, ++iter) {
            matrix.at<float>(i, j) = *iter;
        }
    }

    return matrix;
}

} // namespace face
