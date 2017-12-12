#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>

#include "image4dloader.h"
#include "datasetcov.h"
#include "facerecognizer.h"
#include "image4dloader.h"
#include "preprocessor.h"
#include "test.h"

using std::string;
using std::vector;
using cv::Mat;
using namespace face;

const int SUBSETS = 3;

const cv::String KEYS = "{ help h usage ?      | | print this message }"
                        "{ dataset             | | load images from dataset    }"
                        "{ preprocessedDataset | | load images from preprocessed dataset }"
                        "{ savePreprocessed    | | if set saves preprocessed images }"
                        "{ query               | | path to query identity }"
                        "{ saveTrained         | | path to store trained svms }"
                        "{ loadTrained         | | path to trained svms }";

void testFunctions();
DatasetCov loadAndPreprocess(const string& datasetPath);
bool savePreprocessedDataset(const string& path, const vector<vector<Mat>>& grayscale,
                             const vector<vector<Mat>>& depthmap);

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, KEYS); // opencv parser for command line arguments

    // if wrong arguments, print usage
    if (!parser.has("dataset") && !parser.has("preprocessedDataset")) {
        parser.printMessage();
        return 0;
    }

    //vector<vector<Mat>> grayscale, depthmap;
    DatasetCov dataset;
    if (parser.has("preprocessedDataset")) {
        auto path = parser.get<string>("preprocessedDataset");
        // dataset = DatasetCov::load(path);
    }
    else if (parser.has("dataset")) {

        dataset = loadAndPreprocess(parser.get<string>("dataset"));

        if (parser.has("savePreprocessed")) {
            auto path = parser.get<string>("savePreprocessed");
            auto success = dataset.save(path);
            if (!success)
                std::cerr << "Error saving preprocessed dataset to " << path << std::endl;
        }
    }

    FaceRecognizer faceRec(SUBSETS);
    if (!parser.has("loadTrained")) {
        faceRec.train(dataset.grayscale, dataset.depthmap);
    }
    else {
        faceRec.load(parser.get<string>("loadTrained"));
    }

    if (parser.has("saveTrained")) {
        faceRec.save(parser.get<string>("saveTrained"));
    }

    // prediction

    return 0;
}


DatasetCov loadAndPreprocess(const string& datasetPath)
{
    Image4DLoader loader(datasetPath);
    loader.setFileNameRegEx("frame_[0-9]*_(rgb|depth).*");

    Preprocessor preproc;
    vector<vector<Mat>> grayscale, depthmap;
    for (int i = 1; i < 25; ++i) {
        std::cout << "Identity " << i << std::endl;
        auto path = datasetPath + "/" + (i < 10 ? "0" : "") + std::to_string(i);
        loader.setCurrentPath(path);

        std::cout << "Loading and preprocessing images..." << std::endl;
        auto preprocessdFaces = preproc.preprocess(loader.get());

        std::cout << "Computing covariance representation..." << std::endl;
        vector<Mat> grayscaleCovar, depthmapCovar;
        covariance::getNormalizedCovariances(preprocessdFaces, SUBSETS, grayscaleCovar, depthmapCovar);
        grayscale.push_back(std::move(grayscaleCovar));
        depthmap.push_back(std::move(depthmapCovar));
    }

    return {grayscale, depthmap};
}


void testFunctions()
{
    //test::testSingletonSettings();
    //
    //face::test::testImage4DLoader();
    //
    //test::testFindThreshold();
    //
    //test::testGetDepthMap();
    //
    //test::testKmeans();
    //
    //face::test::testPreprocessing();
    //
    //test::testLoadSpeed();
    //
    //test::testEulerAnglesToRotationMatrix();
    //
    //test::testPoseClustering();
    //
    //test::testKmeans2();
    //
    //face::test::covarianceTest();
    //
    face::test::testSVM();
    //
    //face::test::testSVMLoad();
    //
    //face::test::covarianceTest();
    //
    //face::test::testBackgroundRemoval();

    std::cout << "\n\nTests finished!" << std::endl;
}
