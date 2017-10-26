#include "svmtrainer.h"

#include <experimental/filesystem>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "covariancecomputer.h"
#include <image4dcomponent.h>

using cv::Mat;
using std::string;
using std::vector;

namespace fs = std::experimental::filesystem;

namespace face {

SVMTrainer::SVMTrainer()
{
}

void SVMTrainer::train(Image4DComponent* trainingSamples)
{
    N = trainingSamples->size();

    grayscaleSVMs.resize(N);
    depthmapSVMs.resize(N);
    for (auto& svmVector : grayscaleSVMs)
        svmVector.resize(c);
    for (auto& svmVector : depthmapSVMs)
        svmVector.resize(c);

    //  transform trainingSamples to feature vectors for the SVMs
    MatSet grayscaleCovar, depthmapCovar;
    for (Image4DComponent* imageC1 : *trainingSamples) {
        vector<Mat> grayScaleVec;
        vector<Mat> depthVec;
        for (Image4DComponent* imageC2 : *imageC1) {
            grayScaleVec.push_back(imageC2->getImageCovariance());
            depthVec.push_back(imageC2->getDepthCovariance());
        }
        grayscaleCovar.push_back(grayScaleVec);
        depthmapCovar.push_back(depthVec);
    }

    // convert data format to be ready for SVMs, i.e. from Mat vector to Mat
    vector<Mat> grayscaleMat = formatDataForTraining(grayscaleCovar);
    vector<Mat> depthmapMat = formatDataForTraining(depthmapCovar);

    // train SVMs for grayscale images
    //    trainSVMs(grayscaleMat, grayscaleIndexes, ImgType::grayscale);

    // train SVMs for depthmap images
    trainSVMs(depthmapMat, ImgType::depthmap);
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

void SVMTrainer::trainSVMs(vector<Mat>& data, ImgType svmToTrain)
{
    SVMSteinMatrix& svms = (svmToTrain == ImgType::grayscale) ? grayscaleSVMs : depthmapSVMs;
    for (int i = 0; i < data.size(); ++i) {
        Mat cluster = data[i];
        vector<int> labels(cluster.rows - c + 1, -1);
        for (auto id = 0; id < cluster.rows; ++id) {

            std::cout << "id: " << id << ", subset: " << i << std::endl;
            //            if (indexes[id] + i > data.rows / 2)
            //                targetIndex -= c - 1;
            labels[id] = 1;

            Mat removed;
            Mat trainingData = removeRow(cluster, removed, id);
            svms[id][i].trainAuto(trainingData, labels);
            restoreRow(cluster, removed, id);
            labels[id] = -1;
        }
    }
}

Mat SVMTrainer::removeRow(Mat& data, Mat& removed, int id) const
{
    removed = Mat(1, data.cols, data.type());

    int baseSwapIndex = data.rows - 1;
    cv::Rect roi(0, 0, data.cols, data.rows - 1); // overwrite rows using last n-1 rows.
    if (id == data.rows - 1) { //if it is the last row
        // return the first n-1 rows
        return data(roi);
    }

    // else swap the row with the last one
    data.row(id).copyTo(removed);

    data.row(data.rows - 1).copyTo(data.row(id));
    return data(roi);
}

void SVMTrainer::restoreRow(Mat& data, Mat& removed, int id) const
{
    removed.row(0).copyTo(data.row(id));
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
vector<Mat> SVMTrainer::formatDataForTraining(const MatSet& data)
{

    // compute dataOut dimensions
    int height = 0;
    for (const auto& identity : data) {
        height += identity.size();
    }
    const int width = data[0][0].rows * data[0][0].cols; // assuming all Mat in dataIn have the same dimensions
    vector<Mat> dataOut;

    // every Mat in dataIn is converted to a row of dataOut
    for (size_t i = 0; i < data.size(); ++i) { // for each vector in dataIn (i.e. for each identity)..
        const auto& poseCluster = data[i];
        Mat pose(height, width, data[0][0].type());
        for (size_t j = 0; j < poseCluster.size(); ++j) { // for each Mat belonging to this identity...

            // convert the Mat in a row of DataOut
            auto iter = poseCluster[j].begin<float>();
            for (auto k = 0; k < width; ++k, ++iter) {
                pose.at<float>(j, k) = *iter;
            }
        }
        dataOut.push_back(pose);
    }

    return dataOut;
}

} // namespace face
