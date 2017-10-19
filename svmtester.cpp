#include "svmtester.h"
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

const string SVMTester::unknownIdentity = "unknown_ID";

bool SVMTester::load(const string& directoryName)
{
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

    return true;
}
SVMTester::SVMTester(int c)
    : SVMManager(c)
{
}

string SVMTester::predict(const vector<Face>& identity) const
{
    const vector<Mat> grayscaleCovar, depthmapCovar;
    getNormalizedCovariances(identity, c, grayscaleCovar, depthmapCovar);
    auto grayscaleData = formatDataForPrediction(grayscaleCovar);
    auto depthmapData = formatDataForPrediction(depthmapCovar);

    // count votes for each identity
    vector<int> votes(N);
    int maxVotes = -1;
    for (auto i = 0; i < N; ++i) {
        int vote = 0;
        for (auto j = 0; j < c; ++j) {
            if (grayscaleSVMs[i][j].predict(grayscaleData.row(j)) == 1)
                ++vote;
            if (depthmapSVMs[i][j].predict(depthmapData.row(j)) == 1)
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
    float maxDistance = std::numeric_limits<float>::min();
    int bestIndex = -1;
    for (auto i : ties) {
        float distance = 0;
        for (auto j = 0; j < c; ++j) {
            distance += grayscaleSVMs[i][j].getDistanceFromHyperplane(grayscaleData.row(j));
            distance += depthmapSVMs[i][j].getDistanceFromHyperplane(depthmapData.row(j));
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

Mat SVMTester::formatDataForPrediction(const vector<Mat>& data)
{
    const int height = data.size();
    const int width = data[0].rows * data[0].cols;
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
}
