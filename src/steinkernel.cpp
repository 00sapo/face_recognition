#include "steinkernel.h"

SteinKernel::SteinKernel(const SvmParams& _params = SvmParams())
{
    params = _params;
}

int SteinKernel::getType() const
{
    return ml::SVM::CUSTOM;
}

void SteinKernel::calc(int vcount, int n, const float* x, const float* y, float* results)
{
    Mat Y = Mat(16, 16, CV_32F, (void*)y);

    for (int i = 0; i < vcount; i++) {
        Mat X = Mat(16, 16, CV_32F, (void*)(x + n * i));
        Mat A;
        add(X, Y, A);
        double s = std::log10(determinant(A * 0.5)) - 0.5 * std::log10(determinant(X * Y));
        results[i] = std::exp(-params.gamma * s);
    }
}
