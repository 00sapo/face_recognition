#include "svmmanager.h"
#include "covariancecomputer.h"

using namespace face;

vector<string> SVMManager::generateLabels(int numOfLabels)
{
    string id = "identity_";
    vector<string> identities;

    int numOfDigits = 0;
    for (int N = numOfLabels; N > 0; N /= 10, ++numOfDigits)
        ; // count number of digits

    for (int i = 0; i < numOfLabels; ++i) {
        std::stringstream stream;
        stream << id << std::setfill('0') << std::setw(numOfDigits) << i; // fixed length identity
        identities.push_back(stream.str());
    }

    return identities;
}

void SVMManager::getNormalizedCovariances(const vector<Face>& identity, int subsets, vector<Mat>& grayscaleCovarOut,
    vector<Mat>& depthmapCovarOut)
{
    grayscaleCovarOut.clear();
    depthmapCovarOut.clear();

    auto pairs = covariance::computeCovarianceRepresentation(identity, subsets);
    for (const auto& pair : pairs) {
        Mat normalizedGrayscale, normalizedDepthmap;
        cv::normalize(pair.first, normalizedGrayscale);
        cv::normalize(pair.second, normalizedDepthmap);
        grayscaleCovarOut.push_back(normalizedGrayscale);
        depthmapCovarOut.push_back(normalizedDepthmap);
    }
}

void SVMManager::getNormalizedCovariances(const FaceMatrix& identities, int subsets, MatMatrix& grayscaleCovarOut,
    MatMatrix& depthmapCovarOut)
{
    grayscaleCovarOut.clear();
    depthmapCovarOut.clear();

    for (const auto& identity : identities) {
        vector<Mat> grayscaleCovar, depthmapCovar;
        getNormalizedCovariances(identity, subsets, grayscaleCovar, depthmapCovar);
        grayscaleCovarOut.push_back(std::move(grayscaleCovar));
        depthmapCovarOut.push_back(std::move(depthmapCovar));
    }
}
