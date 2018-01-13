#include <iostream>
#include <random>
#include <vector>

#include <opencv2/highgui/highgui.hpp>

#include "boost/regex.hpp"
#include "datasetcov.h"
#include "facerecognizer.h"
#include "image4dloader.h"
#include "preprocessor.h"
#include "test.h"
#include <experimental/filesystem>
#include <fstream>

using namespace std::experimental::filesystem;
using namespace boost;
using cv::Mat;
using std::string;
using std::vector;
using namespace face;

const string OPTION_HELP = "help h usage ?";
const string OPTION_DATASET = "dataset";
const string OPTION_SAVE_DATASET_TR = "saveTrainingset";
const string OPTION_SAVE_DATASET_VAL = "saveValidationset";
const string OPTION_LOAD_DATASET_TR = "loadTrainingset";
const string OPTION_LOAD_DATASET_VAL = "loadValidationset";
const string OPTION_QUERY = "query";
const string OPTION_SAVE = "saveTrained";
const string OPTION_LOAD = "loadTrained";
const string OPTION_MAP = "idmap";
const string OPTION_UNKNOWN = "unknown";

const int SUBSETS = 3;

const cv::String KEYS = "{ " + OPTION_HELP + " | | print this message }"
                                             "{ "
    + OPTION_DATASET + " | | load dataset images }"
                       "{ "
    + OPTION_LOAD_DATASET_TR + " | | load images from preprocessed training set }"
                               "{ "
    + OPTION_LOAD_DATASET_VAL + " | | load images from preprocessed validation set }"
                                "{ "
    + OPTION_SAVE_DATASET_TR + " | | if set saves preprocessed training set }"
                               "{ "
    + OPTION_SAVE_DATASET_VAL + " | | if set saves preprocessed validation set }"
                                "{ "
    + OPTION_QUERY + " | | path to query directory }"
                     "{ "
    + OPTION_SAVE + " | | path to store trained svms }"
                    "{ "
    + OPTION_LOAD + " | | path to trained svms }"
                    "{ "
    + OPTION_MAP + " | | path to the id map file path to trained svms }"
                   "{ "
    + OPTION_UNKNOWN + " | | path the unknown file (one id per line) }";

void testFunctions();
void loadAndPreprocess(const string& datasetPath, std::size_t covarianceSubsets, DatasetCov& trainingSet, DatasetCov& validationSet);
bool datasetLoader(const cv::CommandLineParser& parser, DatasetCov& trainingSet, DatasetCov& validationSet);
void testing(const FaceRecognizer& faceRec, const DatasetCov& testingset, const vector<string>& idTestingMap);
void splitTrainValidation(const vector<Face>& dataset, vector<Face>& trainingSet, vector<Face>& validationSet);
bool loadMap(const cv::CommandLineParser& parser, vector<std::string>& idmap, int& total_unknown_ids);

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, KEYS); // opencv parser for command line arguments

    // if wrong arguments, print usage
    cout << "Loading dataset..." << endl;
    DatasetCov trainingSet, validationSet;
    if (!datasetLoader(parser, trainingSet, validationSet)) {
        std::cout << "Wrong input args!" << std::endl;
        parser.printMessage();
        return 0;
    }

    cout << "Training SVMs..." << endl;
    FaceRecognizer faceRec(SUBSETS);
    if (parser.has(OPTION_LOAD)) {
        faceRec.load(parser.get<string>(OPTION_LOAD));
    } else {
        faceRec.train(trainingSet, validationSet);
    }

    if (parser.has(OPTION_SAVE)) {
        faceRec.save(parser.get<string>(OPTION_SAVE));
    }

    cout << "Loading MAP" << endl;
    vector<string> idmap;
    int total_unknown_ids = 0;
    if (!loadMap(parser, idmap, total_unknown_ids))
        return 1;

    cout << "Querying model..." << endl;
    if (parser.has(OPTION_QUERY)) {
        auto queryPath = parser.get<string>(OPTION_QUERY);
        if (!is_directory(queryPath)) {
            std::cout << queryPath << " is not a directory!" << std::endl;
            return 0;
        }

        Image4DLoader loader(queryPath, "frame_[0-9]*_(rgb|depth).*");
        Preprocessor preproc;
        regex expr{ ".*/[0-9][0-9]" };
        int correct = 0;
        int correct_unknown = 0;
        int total_number_of_query = 0;
        for (auto& x : directory_iterator(queryPath)) {
            string path = x.path().string();
            if (is_directory(path) && regex_match(path, expr)) {
                loader.setCurrentPath(path);
                auto faces = preproc.preprocess(loader.get());

                vector<Mat> grayscale, depthmap;
                covariance::getNormalizedCovariances(faces, SUBSETS, grayscale, depthmap);

                string predicted = faceRec.predict(grayscale, depthmap);

                std::cout << "Path ";
                if (predicted == x.path().filename()) {
                    std::cout << x.path().filename();
                    correct++;
                } else if (predicted == idmap[std::stoi(x.path().filename())]) {
                    std::cout << idmap[std::stoi(x.path().filename())];
                    correct++;
                } else {
                    std::cout << x.path().filename();
                }

                if (predicted == "unknown" && idmap[std::stoi(x.path().filename())] == "unknown")
                    correct_unknown++;

                std::cout << " predicted ID: "
                          << predicted << std::endl;
                total_number_of_query++;
            }
        }
        std::cout << "-------------------------" << std::endl;
        std::cout << "Rank-1: " << (float)correct / total_number_of_query << std::endl;
        std::cout << "FP-precision: " << (float)correct_unknown / total_unknown_ids << std::endl;
    }

    //testFunctions();

    return 0;
}

bool loadMap(const cv::CommandLineParser& parser, vector<string>& idmap, int& total_unknown_ids)
{
    // loading map
    std::ifstream infile(parser.get<string>(OPTION_MAP));
    std::string line;
    int number_of_ids = 0;
    // counting lines
    while (std::getline(infile, line))
        ++number_of_ids;
    idmap = vector<string>(number_of_ids);
    infile.clear();
    infile.seekg(0, std::ios::beg);

    while (std::getline(infile, line)) {
        if (line.at(0) == '#')
            continue;
        std::istringstream iss(line);
        int a;
        string b;
        if (!(iss >> a >> b)) {
            std::cout << "idmap not well formatted" << std::endl;
            return false;
            break;
        }

        idmap[a] = b;
    }
    infile.close();

    // loading unknown
    infile = std::ifstream(parser.get<string>(OPTION_UNKNOWN));
    while (std::getline(infile, line)) {
        if (line.at(0) == '#')
            continue;
        std::istringstream iss(line);
        int a;
        if (!(iss >> a)) {
            std::cout << "unkown map not well formatted" << std::endl;
            return false;
            break;
        }

        total_unknown_ids++;
        idmap[a] = "unknown";
    }
    infile.close();

    return true;
}
bool datasetLoader(const cv::CommandLineParser& parser, DatasetCov& trainingSet, DatasetCov& validationSet)
{
    if (parser.has(OPTION_LOAD_DATASET_TR) && parser.has(OPTION_LOAD_DATASET_VAL)) {
        auto pathTr = parser.get<string>(OPTION_LOAD_DATASET_TR);
        auto pathVal = parser.get<string>(OPTION_LOAD_DATASET_VAL);

        trainingSet.load(pathTr);
        validationSet.load(pathVal);
    } else if (parser.has(OPTION_DATASET)) {
        auto path = parser.get<string>(OPTION_DATASET);
        loadAndPreprocess(path, SUBSETS, trainingSet, validationSet);
    } else {
        return false;
    }

    if (!trainingSet.isConsistent() || !validationSet.isConsistent())
        std::cout << "Warning! Loaded inconsistent dataset!" << std::endl;
    if (trainingSet.empty() || validationSet.empty()) {
        std::cout << "Error! Loaded empty dataset!" << std::endl;
        return false;
    }

    if (parser.has(OPTION_SAVE_DATASET_TR)) {
        auto path = parser.get<string>(OPTION_SAVE_DATASET_TR);
        auto success = trainingSet.save(path);
        if (!success)
            std::cerr << "Error saving preprocessed dataset to " << path << std::endl;
    }
    if (parser.has(OPTION_SAVE_DATASET_VAL)) {
        auto path = parser.get<string>(OPTION_SAVE_DATASET_VAL);
        auto success = validationSet.save(path);
        if (!success)
            std::cerr << "Error saving preprocessed dataset to " << path << std::endl;
    }

    return true;
}

void loadAndPreprocess(const string& datasetPath, std::size_t covarianceSubsets, DatasetCov& trainingSet, DatasetCov& validationSet)
{
    if (!is_directory(datasetPath)) {
        std::cerr << "ERROR! " << datasetPath << " is not a directory!" << std::endl;
        trainingSet.clear();
        validationSet.clear();
        return;
    }

    Image4DLoader loader(datasetPath, "frame_[0-9]*_(rgb|depth).*");

    Preprocessor preproc;
    vector<vector<Mat>> grayscaleTr, depthmapTr;
    vector<vector<Mat>> grayscaleVal, depthmapVal;
    vector<string> dirMap;
    regex expr{ ".*/[0-9][0-9]" };

    for (auto& x : directory_iterator(datasetPath)) {
        auto path = x.path();
        if (is_directory(path) && regex_match(path.string(), expr)) {
            loader.setCurrentPath(path);

            std::cout << "Loading and preprocessing images from " << path << std::endl;
            auto preprocessedFaces = preproc.preprocess(loader.get());

            vector<Face> train, validation;
            splitTrainValidation(preprocessedFaces, train, validation);

            std::cout << "Computing covariance representation..." << std::endl;
            vector<Mat> grayscaleCovarTr, depthmapCovarTr;
            vector<Mat> grayscaleCovarVal, depthmapCovarVal;

            covariance::getNormalizedCovariances(train, covarianceSubsets, grayscaleCovarTr, depthmapCovarTr);
            covariance::getNormalizedCovariances(validation, covarianceSubsets, grayscaleCovarVal, depthmapCovarVal);

            grayscaleTr.push_back(std::move(grayscaleCovarTr));
            depthmapTr.push_back(std::move(depthmapCovarTr));
            grayscaleVal.push_back(std::move(grayscaleCovarVal));
            depthmapVal.push_back(std::move(depthmapCovarVal));
            dirMap.push_back(path.filename());
        }
    }

    trainingSet = DatasetCov(std::move(grayscaleTr), std::move(depthmapTr), dirMap);
    validationSet = DatasetCov(std::move(grayscaleVal), std::move(depthmapVal), dirMap);
}

/**
 * @brief randomly splits a Face vector into two vectors
 * @param dataset
 * @param trainingSet: two-thirds of the dataset
 * @param validationSet: one-third of the dataset
 */
void splitTrainValidation(const vector<Face>& dataset, vector<Face>& trainingSet, vector<Face>& validationSet)
{
    const int size = dataset.size();
    vector<int> index(size);
    for (auto i = 0; i < size; ++i)
        index[i] = i;

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

    for (auto i = 0; i < size / 3; ++i) {
        int rndIndex = std::uniform_int_distribution<>(0, index.size() - 1)(gen);
        validationSet.push_back(dataset[index[rndIndex]]);

        index[rndIndex] = index.back();
        index.pop_back();
    }

    for (auto i : index) {
        trainingSet.push_back(dataset[i]);
    }
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
