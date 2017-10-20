#include "svmtrainer.h"

#include <experimental/filesystem>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "covariancecomputer.h"
#include "face.h"

using cv::Mat;
using std::string;
using std::vector;

namespace fs = std::experimental::filesystem;

namespace face {

SVMTrainer::SVMTrainer(int c)
    : SVMManager(c)
{
}

void SVMTrainer::train(const FaceMatrix& trainingSamples, const vector<string>& samplIDs)
{
    N = trainingSamples.size();

    // if not enough IDs automatically generate them
    IDs = (N > samplIDs.size()) ? generateLabels(N) : samplIDs;

    grayscaleSVMs.resize(N);
    depthmapSVMs.resize(N);
    for (auto& svmVector : grayscaleSVMs)
        svmVector.resize(c);
    for (auto& svmVector : depthmapSVMs)
        svmVector.resize(c);

    // compute normalized covariances, i.e. transform trainingSamples to feature vectors for the SVMs
    MatMatrix grayscaleCovar, depthmapCovar;
    getNormalizedCovariances(trainingSamples, c, grayscaleCovar, depthmapCovar);

    // convert data format to be ready for SVMs, i.e. from Mat vector to Mat
    vector<int> grayscaleIndexes, depthmapIndexes;
    auto grayscaleMat = formatDataForTraining(grayscaleCovar, grayscaleIndexes);
    auto depthmapMat = formatDataForTraining(depthmapCovar, depthmapIndexes);

    // train SVMs for grayscale images
    //    trainSVMs(grayscaleMat, grayscaleIndexes, ImgType::grayscale);

    // train SVMs for depthmap images
    trainSVMs(depthmapMat, depthmapIndexes, ImgType::depthmap);
}

bool SVMTrainer::save(const string& directoryName)
{
    // create root directory
    fs::path rootDir(directoryName);
    if (!fs::create_directory(rootDir)) {
        std::cerr << "Unable to create directory " << directoryName << std::endl;
        return false;
    }

    for (auto i = 0; i < N; ++i) { // for each identity...
        // create its own subdir
        auto subDir = rootDir / IDs[i];
        if (!fs::create_directory(subDir)) {
            std::cerr << "Unable to create directory " << directoryName << std::endl;
            return false;
        }

        for (auto j = 0; j < c; ++j) { // for each head rotation subset...
            // save svm
            std::stringstream grayscaleStr, depthmapStr;
            grayscaleStr << "grayscale_" << std::setfill('0') << std::setw(3) << j; // fixed length identity
            depthmapStr << "depthmap_" << std::setfill('0') << std::setw(3) << j; // fixed length identity
            string grayscaleFileName = (subDir / grayscaleStr.str()).string();
            string depthmapFileName = (subDir / depthmapStr.str()).string();
            grayscaleSVMs[i][j].save(grayscaleFileName);
            depthmapSVMs[i][j].save(depthmapFileName);
        }
    }

    return true;
}

void SVMTrainer::trainSVMs(Mat& data, const vector<int>& indexes, ImgType svmToTrain)
{
    SVMSteinMatrix& svms = (svmToTrain == ImgType::grayscale) ? grayscaleSVMs : depthmapSVMs;
    for (auto id = 0; id < indexes.size(); ++id) {
        vector<int> labels(data.rows - c + 1, -1);
        for (auto i = 0; i < c; ++i) {
            std::cout << "id: " << id << ", subset: " << i << std::endl;
            int targetIndex = indexes[id] + i;
            if (indexes[id] + i > data.rows / 2)
                targetIndex -= c - 1;
            labels[targetIndex] = 1;

            Mat removed;
            auto trainingData = removeRows(data, removed, indexes[id], i);
            svms[id][i].trainAuto(trainingData, labels);
            restoreRows(data, removed, indexes[id], i);
            labels[targetIndex] = -1;
        }
    }
}

Mat SVMTrainer::removeRows(Mat& data, Mat& removed, int baseIdIndex, int subset) const
{
    removed = Mat(c - 1, data.cols, data.type());

    auto rowToKeep = baseIdIndex + subset;

    int baseSwapIndex = data.rows - c + 1; // if rowToKeep is in the top half of data matrix
    cv::Rect roi(0, 0, data.cols, data.rows - c + 1); // overwrite rows using last c-1 rows.
    if (rowToKeep > data.rows / 2) { // if rowToKeep is in the bottom half of data matrix
        baseSwapIndex = 0; // overwrite rows using first c-1 rows.
        roi.y = c - 1;
    }

    for (auto i = 0; i < subset; ++i) {
        auto rowIndex = baseIdIndex + i; // save row above row id*c + subset
        auto rowToWrite = baseSwapIndex + i; // and overwrite it with rowToWrite
        for (int j = 0; j < removed.cols; ++j) {
            removed.at<float>(i, j) = data.at<float>(rowIndex, j);
            data.at<float>(rowIndex, j) = data.at<float>(rowToWrite, j);
        }
    }

    for (auto i = subset + 1; i < c; ++i) {
        auto rowToSave = baseIdIndex + i; // save row below row id*c + subset
        auto rowToWrite = baseSwapIndex + i - 1; // and overwrite it with rowToWrite
        for (int j = 0; j < removed.cols; ++j) {
            // i-1 because we skipped a row in data but not in removed
            removed.at<float>(i - 1, j) = data.at<float>(rowToSave, j);
            data.at<float>(rowToSave, j) = data.at<float>(rowToWrite, j);
        }
    }

    return data(roi);
}

void SVMTrainer::restoreRows(Mat& data, Mat& removed, int baseIdIndex, int subset) const
{
    for (auto i = 0; i < subset; ++i) {
        auto rowIndex = baseIdIndex + i;
        for (int j = 0; j < removed.cols; ++j) // restore previously removed row
            data.at<float>(rowIndex, j) = removed.at<float>(i, j);
    }

    for (auto i = subset + 1; i < c; ++i) {
        auto rowIndex = baseIdIndex + i;
        for (int j = 0; j < removed.cols; ++j) // restore previously removed row
            // i-1 because we skipped a row in rows left but not in removed
            data.at<float>(rowIndex, j) = removed.at<float>(i - 1, j);
    }
}

/**
 * @brief formatDataForTraining transforms the input dataset in a suitable format to be used by
 *        grayscaleSVMs and depthmapSVMs
 * @param data: vector of identities. For each identity it contains a vector of covariance matrixes
 * @param indexes: a vector with the starting rows indexes for each identity (with length = dataIn.size())
 *
 * @return formatted data to be feed into SVMModel. It has one row for each Mat contained in dataIn
 *         and a number of columns equal to Mat::rows x Mat::columns (assuming every Mat in dataIn
 *         has the same dimensions)
 */
Mat SVMTrainer::formatDataForTraining(const MatMatrix& data, std::vector<int>& indexes)
{
    indexes.clear();

    // compute dataOut dimensions
    int height = 0;
    for (const auto& identity : data) {
        height += identity.size();
    }
    const int width = data[0][0].rows * data[0][0].cols; // assuming all Mat in dataIn have the same dimensions
    Mat dataOut(height, width, data[0][0].type());

    // every Mat in dataIn is converted to a row of dataOut
    int rowIndex = 0;
    for (size_t i = 0; i < data.size(); ++i) { // for each vector in dataIn (i.e. for each identity)..
        const auto& identity = data[i];
        int start = rowIndex; // keep track of the first row index of current identity
        for (size_t j = 0; j < identity.size(); ++j) { // for each Mat belonging to this identity...

            // convert the Mat in a row of DataOut
            auto iter = identity[j].begin<float>();
            for (auto k = 0; k < width; ++k, ++iter) {
                dataOut.at<float>(rowIndex, k) = *iter;
            }
            ++rowIndex;
        }

        // save the range of rows of dataOut belonging to this identity
        indexes.push_back(start);
    }

    return dataOut;
}

} // namespace face
