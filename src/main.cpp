#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>

#include "datasetcov.h"
#include "facerecognizer.h"
#include "image4dloader.h"
#include "preprocessor.h"
#include "test.h"

using cv::Mat;
using std::string;
using std::vector;
using namespace face;

const int SUBSETS = 3;

const cv::String KEYS = "{ help h usage ?           | | print this message }"
                        "{ training                 | | load images from dataset    }"
                        "{ preprocessedTrainingset  | | load images from preprocessed dataset }"
                        "{ saveTrainingset          | | if set saves preprocessed images }"
                        "{ query                    | | path to query identity }"
                        "{ saveTrained              | | path to store trained svms }"
                        "{ loadTrained              | | path to trained svms }"
                        "{ testingset               | | path to testing set }"
                        "{ saveTestingset           | | if set saves preprocessed testing set }"
                        "{ preprocessedTestingset   | | path to preprocessed testing set }";

void testFunctions();
DatasetCov loadAndPreprocess(const string& datasetPath, std::size_t covarianceSubsets);
bool savePreprocessedDataset(const string& path, const vector<vector<Mat>>& grayscale,
    const vector<vector<Mat>>& depthmap);

int proxyDatasetLoader(string type, cv::CommandLineParser parser, DatasetCov dataset)
{
    string optionPreprocessed, optionDataset, optionSave;
    if (type == "training") {
        optionPreprocessed = "preprocessedTrainingset";
        optionDataset = "trainingset";
        optionSave = "saveTrainingset";
    } else if (type == "testing") {
        optionPreprocessed = "preprocessedTestingset";
        optionDataset = "testingset";
        optionSave = "saveTestingset";
    }

    if (parser.has(optionPreprocessed)) {
        auto path = parser.get<string>(optionPreprocessed);

        dataset = DatasetCov::load(path);

        if (!dataset.isConsistent())
            std::cout << "Warning! Loaded inconsistent dataset!" << std::endl;
        if (dataset.empty()) {
            std::cout << "Error! Loaded empty dataset!" << std::endl;
            return 0;
        }
    } else if (parser.has(optionDataset)) {

        dataset = loadAndPreprocess(parser.get<string>(optionDataset), SUBSETS);

        if (parser.has(optionSave)) {
            auto path = parser.get<string>(optionSave);
            auto success = dataset.save(path);
            if (!success)
                std::cerr << "Error saving preprocessed dataset to " << path << std::endl;
        }
    }
}
int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, KEYS); // opencv parser for command line arguments

    // if wrong arguments, print usage
    if (!parser.has("dataset") && !parser.has("preprocessedDataset")) {
        parser.printMessage();
        return 0;
    }

    DatasetCov trainingset, testingset;
    proxyDatasetLoader("training", parser, trainingset);
    proxyDatasetLoader("testing", parser, testingset);

    FaceRecognizer faceRec(SUBSETS);
    if (!parser.has("loadTrained")) {
        faceRec.train(trainingset.grayscale, trainingset.depthmap);
    } else {
        faceRec.load(parser.get<string>("loadTrained"));
    }

    if (parser.has("saveTrained")) {
        faceRec.save(parser.get<string>("saveTrained"));
    }

    // prediction

    //testFunctions();

    return 0;
}

DatasetCov loadAndPreprocess(const string& datasetPath, std::size_t covarianceSubsets)
{
    Image4DLoader loader(datasetPath, "frame_[0-9]*_(rgb|depth).*");

    Preprocessor preproc;
    vector<vector<Mat>> grayscale, depthmap;
    for (int i = 1; i < 25; ++i) { // TODO: load each folder using std::experimental::file_system
        std::cout << "Identity " << i << std::endl;
        auto path = datasetPath + "/" + (i < 10 ? "0" : "") + std::to_string(i);
        loader.setCurrentPath(path);

        std::cout << "Loading and preprocessing images..." << std::endl;
        auto faces = loader.get();
        //for (const auto &face : faces) {
        //    cv::imshow("face",face.image);
        //    cv::waitKey();
        //}
        auto preprocessedFaces = preproc.preprocess(faces);
        std::cout << "Computing covariance representation..." << std::endl;
        vector<Mat> grayscaleCovar, depthmapCovar;
        covariance::getNormalizedCovariances(preprocessedFaces, covarianceSubsets, grayscaleCovar, depthmapCovar);
        grayscale.push_back(std::move(grayscaleCovar));
        depthmap.push_back(std::move(depthmapCovar));
    }

    return { std::move(grayscale), std::move(depthmap) };
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
    face::test::testPreprocessing();
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
    //face::test::testSVM();
    //
    //face::test::testSVMLoad();
    //
    //face::test::covarianceTest();
    //
    //face::test::testBackgroundRemoval();

    std::cout << "\n\nTests finished!" << std::endl;
}
