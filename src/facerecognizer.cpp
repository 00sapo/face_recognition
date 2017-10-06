#include "facerecognizer.h"

#include <iomanip>
#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using cv::Mat;


namespace face {


const string FaceRecognizer::unknownIdentity = "unknown_ID";

FaceRecognizer::FaceRecognizer(int c) : c(c) { }

FaceRecognizer::FaceRecognizer(const string &fileName)
{
    load(fileName);
}

// FIXME: change training to support c head rotations subsets
void FaceRecognizer::train(const FaceMatrix &trainingSamples, const vector<string> &samplIDs)
{
    N = trainingSamples.size();

    // if not enough IDs automatically generate them
    IDs = (N < samplIDs.size()) ? generateLabels(N) : samplIDs;

    grayscaleSVMs.resize(N);
    depthmapSVMs .resize(N);

    // compute normalized covariances, i.e. transform trainingSamples to feature vectors for the SVMs
    MatMatrix grayscaleCovar, depthmapCovar;
    getNormalizedCovariances(trainingSamples, grayscaleCovar, depthmapCovar);

    // convert data format to be ready for SVMs, i.e. from Mat vector to Mat
    vector<cv::Range> grayscaleRanges, depthmapRanges;
    auto grayscaleMat = formatDataForTraining(grayscaleCovar, grayscaleRanges);
    auto depthmapMat  = formatDataForTraining(depthmapCovar,  depthmapRanges);

    // train SVMs for grayscale images
    for (int id = 0; id < grayscaleRanges.size(); ++id) {
        vector<int> labels(grayscaleMat.rows, -1);
        for (auto i = grayscaleRanges[id].start; i < grayscaleRanges[id].end; ++i) {
            labels[i] = 1;
        }

        grayscaleSVMs[id].trainAuto(grayscaleMat, labels);
    }

    // train SVMs for depth images
    for (int id = 0; id < depthmapRanges.size(); ++id) {
        vector<int> labels(depthmapMat.rows, -1);
        for (auto i = depthmapRanges[id].start; i < depthmapRanges[id].end; ++i) {
            labels[i] = 1;
        }

        depthmapSVMs[id].trainAuto(depthmapMat, labels);
    }
}

string FaceRecognizer::predict(const vector<Face> &identity) const
{
    vector<Mat> grayscaleCovar, depthmapCovar;
    getNormalizedCovariances(identity, grayscaleCovar, depthmapCovar);
    auto grayscaleData = formatDataForPrediction(grayscaleCovar);
    auto depthmapData  = formatDataForPrediction(depthmapCovar);


    // count votes for each identity
    vector<int> votes(N);
    int maxVotes = -1, maxIndex = -1;
    for (auto i = 0; i < N; ++i) {
        int vote = 0;
        for (auto j = 0; j < c; ++j) {
            if (grayscaleSVMs[i][j].predict(grayscaleData.row(j)) == 1) ++vote;
            if (depthmapSVMs [i][j].predict(depthmapData .row(j)) == 1) ++vote;
        }
        if (vote > maxVotes) {
            maxVotes = vote;
            maxIndex = i;
        }
        votes[i] = vote;
    }

    // get identities with the same number of votes
    vector<int> ties;
    for (auto i = 0; i < N; ++i) {
        if (votes[i] == maxVotes)
            ties.push_back(i);
    }

    // pick the identity with maximum mean distance from the hyperplane
    float maxDistance = std::numeric_limits<float>::min();
    int bestIndex = -1;
    for (auto i : ties) {
        float distance = 0;
        for (auto j = 0; j < c; ++j) {
            distance += grayscaleSVMs[i][j].getDistanceFromHyperplane(grayscaleData.row(j));
            distance += depthmapSVMs [i][j].getDistanceFromHyperplane(depthmapData .row(j));
        }
        if (distance > maxDistance) {
            maxDistance = distance;
            bestIndex = i;
        }
    }

    if (bestIndex == -1)
        return unknownIdentity;

    return IDs[bestIndex];
}

bool FaceRecognizer::load(const string &fileName)
{

}

bool FaceRecognizer::save(const string &fileName)
{

}


vector<string> FaceRecognizer::generateLabels(int numOfLabels)
{
    string id = "identity_";
    vector<string> identities;

    int numOfDigits = 0;
    for (int N = numOfLabels; N > 0; N /= 10, ++numOfDigits);  // count number of digits

    for (int i = 0; i < numOfLabels; ++i) {
        std::stringstream stream;
        stream << id << std::setfill('0') << std::setw(numOfDigits) << i;   // fixed length identity
        identities.push_back(stream.str());
    }

    return identities;
}


void FaceRecognizer::getNormalizedCovariances(const vector<Face> &identity,
                                              vector<Mat> &grayscaleCovarOut,
                                              vector<Mat> &depthmapCovarOut) const
{
    grayscaleCovarOut.clear();
    depthmapCovarOut .clear();

    auto pairs = covarComputer.computeCovarianceRepresentation(identity, c);
    for (const auto &pair : pairs) {
        Mat normalizedGrayscale, normalizedDepthmap;
        cv::normalize(pair.first , normalizedGrayscale);
        cv::normalize(pair.second, normalizedDepthmap );
        grayscaleCovarOut.push_back(normalizedGrayscale);
        depthmapCovarOut .push_back(normalizedDepthmap );
    }
}


void FaceRecognizer::getNormalizedCovariances(const FaceMatrix &identities,MatMatrix &grayscaleCovarOut,
                                              MatMatrix &depthmapCovarOut) const
{
    grayscaleCovarOut.clear();
    depthmapCovarOut .clear();

    for (const auto& identity : identities) {
        vector<Mat> grayscaleCovar, depthmapCovar;
        getNormalizedCovariances(identity, grayscaleCovar, depthmapCovar);
        grayscaleCovarOut.push_back(std::move(grayscaleCovar));
        depthmapCovarOut .push_back(std::move(depthmapCovar ));
    }
}

Mat FaceRecognizer::formatDataForTraining(const MatMatrix &dataIn, vector<cv::Range> &ranges) const
{
    ranges.clear();

    // compute dataOut dimensions
    int height = 0;
    for (const auto &identity : dataIn) {
        height += identity.size();
    }
    const int width = dataIn[0][0].rows * dataIn[0][0].cols; // assuming all Mat in dataIn have the same dimensions
    Mat dataOut(height, width, dataIn[0][0].type());

    // every Mat in dataIn is converted to a row of dataOut
    int rowIndex = 0;
    for (auto i = 0; i < dataIn.size(); ++i) { // for each vector in dataIn (i.e. for each identity)..
        const auto &identity = dataIn[i];
        int start = rowIndex; // keep track of the first row index of current identity
        for (auto j = 0; j < identity.size(); ++j) { // for each Mat belonging to this identity...

            // convert the Mat in a row of DataOut
            auto iter = identity[j].begin<float>();
            for (auto k = 0; k < width; ++k, ++iter) {
                dataOut.at<float>(rowIndex, k) = *iter;
            }
            ++rowIndex;
        }

        // save the range of rows of dataOut belonging to this identity
        ranges.emplace_back(start, rowIndex);
    }

    return dataOut;
}

Mat FaceRecognizer::formatDataForPrediction(const vector<Mat> &data) const
{
    const int height = data.size();
    const int width  = data[0].rows * data[0].cols;
    Mat dataOut(height, width, data[0].type());

    for (auto i = 0; i < height; ++i) { // for each Mat belonging to this identity...
        // convert the Mat in a row of dataOut
        auto iter = data[i].begin<float>();
        for (auto j = 0; j < width; ++j, ++iter) {
            dataOut.at<float>(i, j) = *iter;
        }
    }

    return dataOut;
}


} // namespace face
