#include "steinkernel.h"

SteinKernel::SteinKernel(const float sigma = 1)
{
    this->sigma = sigma;
}

int SteinKernel::getType() const
{
    return SVM::CUSTOM;
}

void SteinKernel::calc(int vcount, int n, const float* x, const float* y, float* results)
{
    cv::Mat X = cv::Mat(16, 16, CV_32F, (void*)x);
    cv::Mat Y = cv::Mat(16, 16, CV_32F, (void*)y);
    cv::Mat A;
    cv::add(X, Y, A);
    double s = std::log10(cv::determinant(A * 0.5)) - 0.5 * std::log10(cv::determinant(X * Y));
    std::exp(-sigma * s);
}
