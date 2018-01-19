#include "facerecognizer.h"

#include <experimental/filesystem>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "covariancecomputer.h"
#include "datasetcov.h"
#include "face.h"

using cv::Mat;
using std::string;
using std::vector;

namespace fs = std::experimental::filesystem;

namespace face {

// utility functions
vector<string> generateLabels(int numOfLabels);
void getNormalizedCovariances(const vector<Face>& identity, int subsets, vector<Mat>& grayscaleCovarOut,
    vector<Mat>& depthmapCovarOut);
void getNormalizedCovariances(const FaceMatrix& identities, int subsets, MatMatrix& grayscaleCovarOut,
    MatMatrix& depthmapCovarOut);
Mat extractSubset(const Mat& data, int subsetIndex, int totalSubsets);

// --------------------------------------------------
// ------------- FaceRecognizer members -------------
// --------------------------------------------------

FaceRecognizer::FaceRecognizer(int c)
    : c(c)
{
}

FaceRecognizer::FaceRecognizer(const string& fileName)
{
    load(fileName);
}

void FaceRecognizer::train(const DatasetCov& trainingSet, const DatasetCov& validationSet) //const MatMatrix& grayscaleCovar, const MatMatrix& depthmapCovar, const vector<string>& samplIDs)
{
    assert(trainingSet.checkConsistency() && validationSet.checkConsistency());

    N = trainingSet.grayscale.size();

    labels.resize(N);
    for (auto i = 0; i < N; ++i) {
        labels[i] = trainingSet.getDirectory(i);
    }

    grayscaleSVMs.resize(N);
    depthmapSVMs.resize(N);
    for (auto& svmVector : grayscaleSVMs)
        svmVector.resize(c);
    for (auto& svmVector : depthmapSVMs)
        svmVector.resize(c);

    // convert data format to be ready for SVMs, i.e. from Mat vector to Mat
    auto grayscaleMatTr = formatDataForTraining(trainingSet.grayscale);
    auto depthmapMatTr = formatDataForTraining(trainingSet.depthmap);

    vector<int> groundTruthGr, groundTruthDp;
    auto grayscaleMatVal = formatDataForValidation(validationSet.grayscale, groundTruthGr);
    auto depthmapMatVal = formatDataForValidation(validationSet.depthmap, groundTruthDp);

    trainSVMs(grayscaleMatTr, grayscaleMatVal, groundTruthGr, ImgType::grayscale); // grayscale images training
    trainSVMs(depthmapMatTr, depthmapMatVal, groundTruthDp, ImgType::depthmap); // depthmap  images training
}

string FaceRecognizer::predict(const vector<Mat>& grayscaleCovar, const vector<Mat>& depthmapCovar, bool useRGB, bool useDepth) const
{
    auto grayscaleData = formatDataForPrediction(grayscaleCovar);
    auto depthmapData = formatDataForPrediction(depthmapCovar);

    if (!useRGB && !useDepth)
        return "one between useRGB and useDepth must be true";

    // count votes for each identity
    vector<int> votes(N);
    Mat row;
    float prediction;
    int maxVotes = -1;
    int maxRGBVotes = -1;
    int RGBties = 0;
    for (auto i = 0; i < N; ++i) {
        int vote = 0;
        int voteRGB = 0;
        for (auto j = 0; j < c; ++j) {
            if (useRGB) {
                row = grayscaleData.row(j);
                prediction = grayscaleSVMs[i][j].predict(row);
                if (prediction == 1) {
                    ++vote;
                    ++voteRGB;
                }
            }
            if (useDepth) {
                row = depthmapData.row(j);
                prediction = depthmapSVMs[i][j].predict(row);
                if (prediction == 1)
                    ++vote;
            }
        }
        if (voteRGB > maxRGBVotes) {
            maxRGBVotes = voteRGB;
            RGBties = 1;
        } else if (voteRGB == maxRGBVotes) {
            RGBties++;
        }
        if (vote > maxVotes)
            maxVotes = vote;

        votes[i] = vote;
    }

    // get identities with the same number of votes
    vector<int> ties;
    for (auto i = 0; i < N; ++i) {
        if (votes[i] == maxVotes)
            ties.push_back(i);
    }

    // pick the identity with maximum mean distance from the hyperplane
    auto maxDistance = std::numeric_limits<float>::min();
    int bestIndex = -1;
    float dist;
    for (auto i : ties) {
        float distance = 0;
        for (auto j = 0; j < c; ++j) {
            if (useRGB) {
                dist = grayscaleSVMs[i][j].getDistanceFromHyperplane(grayscaleData.row(j));
                if (std::isnormal(dist))
                    distance += dist;
            }
            if (useDepth) {
                dist = depthmapSVMs[i][j].getDistanceFromHyperplane(depthmapData.row(j));
                if (std::isnormal(dist))
                    distance += dist;
            }
        }
        if (distance > maxDistance) {
            maxDistance = distance;
            bestIndex = i;
        }
    }

    if (useDepth && !useRGB)
        RGBties = ties.size();
    if (RGBties > c)
        bestIndex = -1;

    if (bestIndex == -1)
        return "unknown";

    return labels[bestIndex];
}

bool FaceRecognizer::load(const string& directoryName)
{
    /*
    IDs.clear();
    grayscaleSVMs.clear();
    depthmapSVMs.clear();

    int numOfIdentities = 0;
    for (const auto& subdir : fs::directory_iterator(directoryName)) {
        ++numOfIdentities;
        vector<fs::path> dirElements;
        for (const auto& dirElement : fs::directory_iterator(subdir)) {
            dirElements.push_back(dirElement);
        }
        std::sort(dirElements.begin(), dirElements.end());

        vector<SVMStein> graySVMs, depthSVMs;
        for (const auto& elem : dirElements) {
            std::cout << "Loading " << elem << std::endl;
            try {
                if (elem.filename().string().find("grayscale") == 0)
                    graySVMs.emplace_back(elem.string());
                else if (elem.filename().string().find("depthmap") == 0)
                    depthSVMs.emplace_back(elem.string());
                else
                    std::cout << "Unrecognized directory element: " << elem;
            } catch (const cv::Exception& ex) {
                std::cout << ex.what() << std::endl;
            }
        }
        c = dirElements.size();
        grayscaleSVMs.push_back(std::move(graySVMs));
        depthmapSVMs.push_back(std::move(depthSVMs));
        std::cout << "Finished loading identity " << subdir.path().filename().string() << std::endl;
        IDs.push_back(subdir.path().filename().string());
    }

    N = numOfIdentities;
    */
    throw NotImplementedException();
    return true;
}

bool FaceRecognizer::save(const string& directoryName)
{
    /*
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
    */
    throw NotImplementedException();
    return true;
}

void FaceRecognizer::trainSVMs(const Mat& dataTr, const Mat& dataVal, const vector<int>& groundTruth, ImgType svmToTrain)
{
    assert(dataTr.rows == N * c && dataVal.rows == N * c && "data must be an Nxc matrix!");

    auto& svms = (svmToTrain == ImgType::grayscale) ? grayscaleSVMs : depthmapSVMs;
    vector<int> labels(N * c, -1);

    for (auto id = 0; id < N; ++id) { // iterate through identities
        for (auto i = 0; i < c; ++i) { // iterate through subsets
            std::cout << "id: " << id << ", subset: " << i << std::endl;
            labels[id * c + i] = 1;
            svms[id][i].trainAuto(dataTr, labels, dataVal, groundTruth);
            labels[id * c + i] = -1;
        }
    }
}

Mat FaceRecognizer::formatDataForTraining(const MatMatrix& data) const
{
    assert(data.size() == N && data[0].size() == c && "data must be a Nxc matrix!");

    // compute dataOut dimensions
    int height = N * c;
    const int width = data[0][0].rows * data[0][0].cols; // assuming all Mat in dataIn have the same dimensions
    Mat dataOut(height, width, data[0][0].type());

    // every Mat in dataIn is converted to a row of dataOut
    for (size_t i = 0; i < N; ++i) { // for each vector in dataIn (i.e. for each identity)..
        const auto& identity = data[i];
        for (size_t j = 0; j < c; ++j) { // for each Mat belonging to this identity (i.e. for each rotation subset)...
            // convert the Mat in a row of DataOut
            auto iter = identity[j].begin<float>();
            for (auto k = 0; k < width; ++k, ++iter) {
                dataOut.at<float>(i * c + j, k) = *iter;
            }
        }
    }

    return dataOut;
}

Mat FaceRecognizer::formatDataForValidation(const MatMatrix& data, std::vector<int>& groundTruth) const
{
    assert(data.size() == N && data[0].size() == c && "data must be a Nxc matrix!");

    // compute dataOut dimensions
    const int height = N * c;
    const int width = data[0][0].rows * data[0][0].cols; // assuming all Mat in dataIn have the same dimensions
    Mat dataOut(height, width, data[0][0].type());
    groundTruth.clear();
    groundTruth.resize(height);

    // every Mat in dataIn is converted to a row of dataOut
    for (size_t i = 0; i < N; ++i) { // for each vector in dataIn (i.e. for each identity)..
        const auto& identity = data[i];
        for (size_t j = 0; j < c; ++j) { // for each Mat belonging to this identity (i.e. for each rotation subset)...
            // convert the Mat in a row of DataOut
            auto iter = identity[j].begin<float>();
            auto rowIndex = c * i + j;
            for (auto k = 0; k < width; ++k, ++iter) {
                dataOut.at<float>(rowIndex, k) = *iter;
            }
            groundTruth[rowIndex] = i;
        }
    }

    return dataOut;
}

Mat FaceRecognizer::formatDataForPrediction(const vector<Mat>& data) const
{
    assert(data.size() == c && "data must be a 1xc matrix!");

    const int WIDTH = data[0].rows * data[0].cols;
    Mat dataOut(c, WIDTH, data[0].type());

    for (auto i = 0; i < c; ++i) { // for each Mat belonging to this identity...
        // convert the Mat in a row of dataOut
        auto iter = data[i].begin<float>();
        for (auto j = 0; j < WIDTH; ++j, ++iter) {
            dataOut.at<float>(i, j) = *iter;
        }
    }

    return dataOut;
}

// ---------------------------------------
// ---------- Utility functions ----------
// ---------------------------------------

vector<string> generateLabels(int numOfLabels)
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

} // namespace face
