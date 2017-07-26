#ifndef STEINKERNEL_H
#define STEINKERNEL_H
#include "opencv2/ml.hpp"

using std;
using cv::ml::SVM;

class SteinKernel : public SVM::Kernel {

public:
    SteinKernel(const float sigma);

    /**
     * @brief cv::ml::SVM::Kernel::calc compute the stein kernel function
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

    float sigma;
};

#endif // STEINKERNEL_H
