#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>

#include "boost/regex.hpp"
#include "datasetcov.h"
#include "facerecognizer.h"
#include "image4dloader.h"
#include "preprocessor.h"
#include "test.h"
#include <experimental/filesystem>

using namespace std::experimental::filesystem;
using namespace boost;
using cv::Mat;
using std::string;
using std::vector;
using namespace face;

const int SUBSETS = 3;

const cv::String KEYS = "{ help h usage ?           | | print this message }"
                        "{ trainingset              | | load images from training set    }"
                        "{ preprocessedTrainingset  | | load images from preprocessed training set }"
                        "{ saveTrainingset          | | if set saves preprocessed images }"
                        "{ query                    | | path to query identity }"
                        "{ saveTrained              | | path to store trained svms }"
                        "{ loadTrained              | | path to trained svms }"
                        "{ testingset               | | load images from testing set }"
                        "{ saveTestingset           | | if set saves preprocessed testing set }"
                        "{ preprocessedTestingset   | | load images from preprocessed testing set }";

void testFunctions();
DatasetCov loadAndPreprocess(const string& datasetPath, std::size_t covarianceSubsets, vector<string> idMap);
bool savePreprocessedDataset(const string& path, const vector<vector<Mat>>& grayscale,
    const vector<vector<Mat>>& depthmap);
int proxyDatasetLoader(string type, cv::CommandLineParser parser, DatasetCov dataset, vector<string> idMap)
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

        dataset = DatasetCov::load(path, idMap);

        if (!dataset.isConsistent())
            std::cout << "Warning! Loaded inconsistent dataset!" << std::endl;
        if (dataset.empty()) {
            std::cout << "Error! Loaded empty dataset!" << std::endl;
            return 1;
        }
    } else if (parser.has(optionDataset)) {

        dataset = loadAndPreprocess(parser.get<string>(optionDataset), SUBSETS, idMap);

        if (parser.has(optionSave)) {
            auto path = parser.get<string>(optionSave);
            auto success = dataset.save(path);
            if (!success)
                std::cerr << "Error saving preprocessed dataset to " << path << std::endl;
        }
    }
    return 0;
}
void testing(FaceRecognizer faceRec, DatasetCov testingset, vector<string> idTestingMap, vector<string> idTrainingMap)
{
    for (int id = 0; id < testingset.depthmap.size(); id++) {
        string predicted = faceRec.predict(testingset.grayscale[id], testingset.depthmap[id]);
        cout << "testing id: name = " << idTestingMap[id] << " prediction " << predicted << endl;
    }
}

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, KEYS); // opencv parser for command line arguments

    // if wrong arguments, print usage
    if (!parser.has("trainingset") && !parser.has("preprocessedTrainingset")) {
        parser.printMessage();
        return 0;
    }

    vector<string> idTestingMap, idTrainingMap;
    DatasetCov trainingset, testingset;
    proxyDatasetLoader("training", parser, trainingset, idTrainingMap);
    proxyDatasetLoader("testing", parser, testingset, idTestingMap);

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
    testing(faceRec, testingset, idTestingMap, idTrainingMap);

    //testFunctions();

    return 0;
}

DatasetCov loadAndPreprocess(const string& datasetPath, std::size_t covarianceSubsets, vector<string> idMap)
{
    Image4DLoader loader(datasetPath, "frame_[0-9]*_(rgb|depth).*");

    Preprocessor preproc;
    vector<vector<Mat>> grayscale, depthmap;
    vector<string> files;
    regex expr{ ".*/[0-9][0-9]" };
    int id = 0;
    if (is_directory(datasetPath)) {
        for (auto& x : directory_iterator(datasetPath)) {
            string path = x.path().string();
            if (is_directory(path) && regex_match(path, expr)) {
                std::cout << "Identity " << id << std::endl;
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
                idMap.push_back(path.substr(path.length() - 2));
                id++;
            }
        }
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
