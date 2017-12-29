#include "facerecognizer.h"

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

const string FaceRecognizer::unknownIdentity = "unknown_ID";

FaceRecognizer::FaceRecognizer(int c)
    : c(c)
{
}

FaceRecognizer::FaceRecognizer(const string& fileName)
{
    load(fileName);
}

void FaceRecognizer::train(const MatMatrix& grayscaleCovar, const MatMatrix& depthmapCovar, const vector<string>& samplIDs)
{
    assert(grayscaleCovar.size() == depthmapCovar.size());
    N = grayscaleCovar.size();

    // if not enough IDs automatically generate them
    IDs = (N > samplIDs.size()) ? generateLabels(N) : samplIDs;

    grayscaleSVMs.resize(N);
    depthmapSVMs.resize(N);
    for (auto& svmVector : grayscaleSVMs)
        svmVector.resize(c);
    for (auto& svmVector : depthmapSVMs)
        svmVector.resize(c);

    // convert data format to be ready for SVMs, i.e. from Mat vector to Mat
    auto grayscaleMat = formatDataForTraining(grayscaleCovar);
    auto depthmapMat = formatDataForTraining(depthmapCovar);

    trainSVMs(grayscaleMat, ImgType::grayscale); // grayscale images training
    trainSVMs(depthmapMat, ImgType::depthmap); // depthmap  images training
}

string FaceRecognizer::predict(const vector<Mat>& grayscaleCovar, const vector<Mat>& depthmapCovar) const
{
    //vector<Mat> grayscaleCovar, depthmapCovar;
    //getNormalizedCovariances(identity, 1 /*c*/, grayscaleCovar, depthmapCovar);
    auto grayscaleData = formatDataForPrediction(grayscaleCovar);
    auto depthmapData = formatDataForPrediction(depthmapCovar);

    // count votes for each identity
    vector<int> votes(N);
    int maxVotes = -1;
    for (auto i = 0; i < N; ++i) {
        int vote = 0;
        for (auto j = 0; j < c; ++j) {
            auto row = grayscaleData.row(j);
            auto prediction = grayscaleSVMs[i][j].predict(row);
            if (prediction == 1)
                ++vote;
            row = depthmapData.row(j);
            prediction = depthmapSVMs[i][j].predict(row);
            if (prediction == 1)
                ++vote;
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
    for (auto i : ties) {
        float distance = 0;
        for (auto j = 0; j < c; ++j) {
            auto dist = grayscaleSVMs[i][j].getDistanceFromHyperplane(grayscaleData.row(j));
            if (std::isnormal(dist))
                distance += dist;
            dist = depthmapSVMs[i][j].getDistanceFromHyperplane(depthmapData.row(j));
            if (std::isnormal(dist))
                distance += dist;
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

void FaceRecognizer::trainSVMs(const Mat& data, ImgType svmToTrain)
{
    assert(data.rows == N * c && "data must be an Nxc matrix!");

    auto& svms = (svmToTrain == ImgType::grayscale) ? grayscaleSVMs : depthmapSVMs;
    //vector<Mat> trainingData;
    //for (auto i = 0; i < c; ++i)
    //    trainingData.push_back(extractSubset(data, i, c));
    vector<int> labels(N * c, -1);

    for (auto id = 0; id < N; ++id) {
        for (auto i = 0; i < c; ++i) {
            std::cout << "id: " << id << ", subset: " << i << std::endl;
            //for (auto i = 0; i < c; ++i)
            labels[id * c + i] = 1;
            // svms[id].trainAuto(data, labels);
            svms[id][i].trainAuto(data, labels);
            //for (auto i = 0; i < c; ++i)
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

/*
void getNormalizedCovariances(const vector<Face>& identity, int subsets, vector<Mat>& grayscaleCovarOut,
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

void getNormalizedCovariances(const FaceMatrix& identities, int subsets, MatMatrix& grayscaleCovarOut,
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
*/

Mat extractSubset(const Mat& data, int subsetIndex, int totalSubsets)
{
    const auto HEIGHT = data.rows / totalSubsets;
    Mat out(HEIGHT, data.cols, data.type());

    for (auto i = 0; i < HEIGHT; ++i) {
        auto identityIndex = totalSubsets * i;
        for (auto j = 0; j < data.cols; ++j) {
            out.at<float>(i, j) = data.at<float>(identityIndex + subsetIndex, j);
        }
    }

    return out;
}

} // namespace face
