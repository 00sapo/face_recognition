#ifndef SVMPARAMS_H
#define SVMPARAMS_H
#include "opencv2/ml.hpp"

namespace cv {
namespace ml {

    struct SvmParams {
        int svmType;
        int kernelType;
        double gamma;
        double coef0;
        double degree;
        double C;
        double nu;
        double p;
        Mat classWeights;
        TermCriteria termCrit;
    };
}
}

#endif // SVMPARAMS_H
