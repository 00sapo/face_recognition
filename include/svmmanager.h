#ifndef SVMMANAGER_H
#define SVMMANAGER_H
#include <opencv2/opencv.hpp>
#include <string>
#include <svmstein.h>
#include <vector>

using std::string;
using std::vector;
using namespace cv;

namespace face {

using FaceMatrix = std::vector<std::vector<face::Face>>;
using MatSet = std::vector<std::vector<cv::Mat>>;
typedef std::vector<std::vector<SVMStein>> SVMSteinMatrix;

class SVMManager {
public:
    SVMManager(int c = 3);

protected:
    // utility functions
    vector<string> generateLabels(int numOfLabels);
    void getNormalizedCovariances(const vector<Face>& identity, int subsets, vector<Mat>& grayscaleCovarOut,
        vector<Mat>& depthmapCovarOut) const;
    void getNormalizedCovariances(const FaceMatrix& identities, int subsets, MatSet& grayscaleCovarOut,
        MatSet& depthmapCovarOut) const;

    enum class ImgType {
        grayscale,
        depthmap
    };

    int c = 3; // number of head rotation subsets for each identity
    int N = 0; // number of identities provided for training
    std::vector<std::string> IDs; // labels associated to each identity in the same order as in grayscaleSVMs and depthmapSVMs
    SVMSteinMatrix grayscaleSVMs; // a row for each identity and a column for each head rotation subset
    SVMSteinMatrix depthmapSVMs; // thus resulting in a Nxc matrix where N is the number of identities
        // and c the number of head rotation subsets
};
}
#endif // SVMMANAGER_H
