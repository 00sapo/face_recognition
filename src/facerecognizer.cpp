#include "facerecognizer.h"

#include <iomanip>
#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using cv::Mat;

namespace face {


const string FaceRecognizer::unknownIdentity = "unknown_ID";

FaceRecognizer::FaceRecognizer() { }

void FaceRecognizer::train(const vector<vector<Face>> &trainingSamples, const vector<string> &samplIDs)
{
    const auto N = trainingSamples.size();

    // if not enough IDs automatically generate them
    IDs = (N < samplIDs.size()) ? generateLabels(N) : samplIDs;

    grayscaleSVMs.reserve(N);
    depthmapSVMs .reserve(N);

    // compute normalized covariances, i.e. transform trainingSamples to feature vectors for the SVMs
    vector<vector<Mat>> grayscaleCovar, depthmapCovar;
    for (const auto& identity : trainingSamples) {
        auto pairs = covarComputer.computeCovarianceRepresentation(identity, c);
        vector<Mat> depthmap, grayscale;
        for (const auto &pair : pairs) {
            Mat normalizedGrayscale, normalizedDepthmap;
            cv::normalize(pair.first , normalizedGrayscale);
            cv::normalize(pair.second, normalizedDepthmap );
            grayscale.push_back(normalizedGrayscale);
            depthmap .push_back(normalizedDepthmap );
        }
        grayscaleCovar.push_back(std::move(grayscale));
        depthmapCovar .push_back(std::move(depthmap ));
    }



    for (const auto &identity : grayscaleCovar) {

    }

    //std::cout << "Creating SVM model..." << std::endl;
    //SVMModel model;
    //std::cout << "Training model..." << std::endl;
    //auto optimalParams = model.trainAuto(person, others);
    //std::cout << "Done!" << std::endl;
    //std::cout << "C: " << optimalParams.C << std::endl;
    //std::cout << "gamma: " << optimalParams.gamma << std::endl;
}

string FaceRecognizer::predict(const vector<Face> &identity) const
{

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

vector<cv::Range> FaceRecognizer::formatDataForTraining(const vector<vector<Mat>> &dataIn, Mat &dataOut) const
{
    vector<cv::Range> ranges;

    // compute dataOut dimensions
    int height = 0;
    for (const auto &identity : dataIn) {
        height += identity.size();
    }
    const int width = dataIn[0][0].rows * dataIn[0][0].cols; // assuming all Mat in dataIn have the same dimensions
    dataOut = Mat(height, width, dataIn[0][0].type());

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

    return ranges;
}


} // namespace face
