#ifndef STEINKERNEL_H
#define STEINKERNEL_H
#include "opencv2/ml.hpp"
#include "svmparams.h"

using namespace cv;
using namespace cv::ml;

class SteinKernel : public SVM::Kernel {

public:
    SteinKernel(const SvmParams& _params);

    /**
     * @brief cv::ml::SVM::Kernel::calc compute the stein kernel function. It uses only gamma parameters from cv::ml::SvmParams
     * @param vcount number of samples (matrixes in our case)
     * @param n length of sample (# of matrixes elements in our case)
     * @param x one of the input sample
     * @param y another input sample
     * @param results array of the results
     */
    void calc(int vcount,
        int n,
        const float* x,
        const float* y,
        float* results);

    int getType() const;
    SvmParams params;
};

#endif // STEINKERNEL_H
