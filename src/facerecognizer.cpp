#include "facerecognizer.h"

#include <iomanip>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

using std::vector;
using std::string;

namespace fs = std::experimental::filesystem;

using cv::Mat;


namespace face {


vector<string> generateLabels(int numOfLabels);
Mat formatDataForTraining  (const MatMatrix &dataIn, vector<int> &indexes);
Mat formatDataForPrediction(const vector<Mat> &data);


// -----------------------------------------------------
// ---------- FaceRecognizer member functions ----------
// -----------------------------------------------------

const string FaceRecognizer::unknownIdentity = "unknown_ID";

FaceRecognizer::FaceRecognizer(int c) : c(c) { }

FaceRecognizer::FaceRecognizer(const string &fileName)
{
    load(fileName);
}


void FaceRecognizer::train(const FaceMatrix &trainingSamples, const vector<string> &samplIDs)
{
    N = trainingSamples.size();

    // if not enough IDs automatically generate them
    IDs = (N > samplIDs.size()) ? generateLabels(N) : samplIDs;

    grayscaleSVMs.resize(N);
    depthmapSVMs .resize(N);
    for (auto &svmVector : grayscaleSVMs) svmVector.resize(c);
    for (auto &svmVector : depthmapSVMs ) svmVector.resize(c);

    // compute normalized covariances, i.e. transform trainingSamples to feature vectors for the SVMs
    MatMatrix grayscaleCovar, depthmapCovar;
    getNormalizedCovariances(trainingSamples, grayscaleCovar, depthmapCovar);

    // convert data format to be ready for SVMs, i.e. from Mat vector to Mat
    vector<int> grayscaleIndexes, depthmapIndexes;
    auto grayscaleMat = formatDataForTraining(grayscaleCovar, grayscaleIndexes);
    auto depthmapMat  = formatDataForTraining(depthmapCovar,  depthmapIndexes);


    // train SVMs for grayscale images

    /*
     * TODO: When training svm[id][i] all head rotation subsets for identity id
     *       are considered, giving label -1 to all but subset i. A better approach
     *       would exclude all subsets of identity id that are not subset i.
     *       In this way we have a better chance of avoiding overfitting (remember
     *       we are given only 1 positive sample).
     */
    for (int id = 0; id < N; ++id) {
        vector<int> labels(grayscaleMat.rows, -1);
        for (auto i = 0; i < c; ++i) {
            int matrixRow = i + grayscaleIndexes[id];
            labels[matrixRow] = 1;
            grayscaleSVMs[id][i].trainAuto(grayscaleMat, labels);
            labels[matrixRow] = -1;
        }
    }

    // train SVMs for depth images
    for (int id = 0; id < N; ++id) {
        vector<int> labels(depthmapMat.rows, -1);
        for (auto i = 0; i < c; ++i) {
            int matrixRow = i + depthmapIndexes[id];
            labels[matrixRow] = 1;
            depthmapSVMs[id][i].trainAuto(depthmapMat, labels);
            labels[matrixRow] = -1;
        }
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
    int maxVotes = -1;
    for (auto i = 0; i < N; ++i) {
        int vote = 0;
        for (auto j = 0; j < c; ++j) {
            if (grayscaleSVMs[i][j].predict(grayscaleData.row(j)) == 1) ++vote;
            if (depthmapSVMs [i][j].predict(depthmapData .row(j)) == 1) ++vote;
        }
        if (vote > maxVotes) maxVotes = vote;

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

bool FaceRecognizer::load(const string &directoryName)
{
    IDs.clear();
    grayscaleSVMs.clear();
    depthmapSVMs .clear();

    int numOfIdentities = 0;
    for (const auto &subdir : fs::directory_iterator(directoryName)) {
        ++numOfIdentities;
        vector<fs::path> dirElements;
        for (const auto &dirElement : fs::directory_iterator(subdir)) {
            dirElements.push_back(dirElement);
        }
        std::sort(dirElements.begin(), dirElements.end());

        vector<SVMStein> graySVMs, depthSVMs;
        for (const auto &elem : dirElements) {
            std::cout << "Loading " << elem << std::endl;
            try {
            if (elem.filename().string().find("grayscale") == 0)
                graySVMs.emplace_back(elem.string());
            else if (elem.filename().string().find("depthmap") == 0)
                depthSVMs.emplace_back(elem.string());
            else
                std::cout << "Unrecognized directory element: " << elem;
            }
            catch (const cv::Exception &ex) {
                std::cout << ex.what() << std::endl;
            }
        }
        c = dirElements.size();
        grayscaleSVMs.push_back(std::move(graySVMs ));
        depthmapSVMs .push_back(std::move(depthSVMs));
        std::cout << "Finished loading identity " << subdir.path().filename().string() << std::endl;
        IDs.push_back(subdir.path().filename().string());
    }

    N = numOfIdentities;
}

bool FaceRecognizer::save(const string &directoryName)
{
    // create root directory
    fs::path rootDir(directoryName);
    if (!fs::create_directory(rootDir)) {
        std::cerr << "Unable to create directory " << directoryName << std::endl;
        return false;
    }

    for (auto i = 0; i < N; ++i) {  // for each identity...
        // create its own subdir
        auto subDir = rootDir / IDs[i];
        if (!fs::create_directory(subDir)) {
            std::cerr << "Unable to create directory " << directoryName << std::endl;
            return false;
        }

        for (auto j = 0; j < c; ++j) {  // for each head rotation subset...
            // save svm
            std::stringstream grayscaleStr, depthmapStr;
            grayscaleStr << "grayscale_" << std::setfill('0') << std::setw(3) << j;   // fixed length identity
            depthmapStr  << "depthmap_"  << std::setfill('0') << std::setw(3) << j;   // fixed length identity
            string grayscaleFileName = (subDir / grayscaleStr.str()).string();
            string depthmapFileName  = (subDir / depthmapStr .str()).string();
            grayscaleSVMs[i][j].save(grayscaleFileName);
            depthmapSVMs [i][j].save(depthmapFileName );
        }
    }

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




// ---------------------------------------
// ---------- Utility functions ----------
// ---------------------------------------

vector<string> generateLabels(int numOfLabels)
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


/**
 * @brief formatDataForTraining transforms the input dataset in a suitable format to be used by
 *        grayscaleSVMs and depthmapSVMs
 * @param dataIn: vector of identities. For each identity it contains a vector of covariance matrixes
 * @param dataOut: formatted data to be feed into SVMModel. It has one row for each Mat contained in dataIn
 *        and a number of columns equal to Mat::rows x Mat::columns (assuming every Mat in dataIn
 *        has the same dimensions)
 * @return a vector with the ranges of rows belonging to each identity (with length = dataIn.size())
 */
Mat formatDataForTraining(const MatMatrix &dataIn, std::vector<int> &indexes)
{
    indexes.clear();

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
        indexes.push_back(start);
    }

    return dataOut;
}

Mat formatDataForPrediction(const vector<Mat> &data)
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
