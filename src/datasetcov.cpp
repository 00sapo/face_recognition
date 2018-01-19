#include "datasetcov.h"

#include <iostream>
#include <regex>

#include <opencv2/highgui.hpp>

using cv::Mat;
using std::string;
using std::vector;

namespace face {

cv::Mat encode(const cv::Mat& image);
cv::Mat decode(const cv::Mat& image);

DatasetCov::DatasetCov()
    : consistent(true)
{
}

DatasetCov::DatasetCov(vector<vector<Mat>> grayscale, vector<vector<Mat>> depthmap, std::vector<std::string> directoryMap)
    : grayscale(std::move(grayscale))
    , depthmap(std::move(depthmap))
    , directoryMap(std::move(directoryMap))
{
    consistent = checkConsistency();
}

std::string DatasetCov::getDirectory(int id) const
{
    return directoryMap.at(id);
}

bool DatasetCov::empty() const
{
    return grayscale.empty() && depthmap.empty();
}

void DatasetCov::clear()
{
    grayscale.clear();
    depthmap.clear();
    directoryMap.clear();
}

bool DatasetCov::isConsistent() const
{
    return consistent;
}

bool DatasetCov::checkConsistency() const
{
    if (grayscale.size() != depthmap.size() || depthmap.size() != directoryMap.size())
        return false;

    for (std::size_t i = 0; i < grayscale.size(); ++i) {
        if (grayscale[i].size() != depthmap[i].size())
            return false;
    }

    return true;
}

bool DatasetCov::save(const std::string& path)
{
    bool saved = false;
    fs::path rootDir(path);
    if (fs::exists(rootDir)) {
        bool done = false;
        while (!done) {
            std::cout << rootDir << " already exists. "
                      << "Do you want to delete its content and write the dataset to this folder? [y|n]" << std::endl;
            char answer;
            std::cin >> answer;
            if (answer == 'y') {
                fs::remove_all(rootDir);
                saved = save(rootDir);
                done = true;
            } else if (answer == 'n') {
                std::cout << "Dataset not saved." << std::endl;
                done = true;
            }
        }
    } else {
        saved = save(rootDir);
    }
    return saved;
}

bool DatasetCov::save(const fs::path& path)
{
    assert(checkConsistency() && "Error! Inconsistent dataset!");

    try { // try creating dataset root directory
        if (!fs::create_directory(path))    // if path already exists fs::create_directory(path) returns true
            return false;
    } catch (const fs::filesystem_error& fsex) {
        std::cerr << fsex.what() << std::endl;
        return false;
    }

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    bool success = true;
    for (std::size_t i = 0; i < grayscale.size(); ++i) { // for each identity
        const auto& grayscaleID = grayscale[i];
        const auto& depthmapID = depthmap[i];

        auto idPath = path / directoryMap[i];

        try { // try creating identity's directory
            if (!fs::create_directory(idPath))
                return false;
        } catch (const fs::filesystem_error& fsex) {
            std::cerr << fsex.what() << std::endl;
            return false;
        }

        for (std::size_t j = 0; j < grayscaleID.size(); ++j) { // save Mats
            if (grayscaleID[j].empty() || depthmapID[j].empty())
                std::cout << "Warning! Trying to save empty image!" << std::endl;

            auto grayscalePath = idPath / ("grayscale_" + std::to_string(j) + ".png");
            auto depthmapPath = idPath / ("depthmap_" + std::to_string(j) + ".png");

            auto grayscaleImg = encode(grayscaleID[j]);
            auto depthmapImg = encode(depthmapID[j]);

            success &= cv::imwrite(grayscalePath.string(), grayscaleImg, compression_params);
            success &= cv::imwrite(depthmapPath.string(),  depthmapImg,  compression_params);
        }
    }

    return success;
}

void DatasetCov::load(const std::string& path)
{
    fs::path datasetPath(path);
    if (!fs::exists(datasetPath)) {
        std::cout << "Warning! " << path << " not found." << std::endl;
    }

    if (!fs::is_directory(datasetPath)) {
        std::cout << "Warning! " << path << " is not a directory." << std::endl;
    }

    std::regex grayscaleTemplate("grayscale_.*png");
    std::regex depthmapTemplate("depthmap_.*png");
    for (const auto& subdir : fs::directory_iterator(datasetPath)) {
        vector<string> grayscaleFiles, depthmapFiles;

        for (const auto& dirEntry : fs::directory_iterator(subdir)) {
            auto file = dirEntry.path();
            auto fileName = file.filename();
            if (std::regex_match(fileName.string(), grayscaleTemplate, std::regex_constants::match_any)) {
                grayscaleFiles.push_back(file.string());
            } else if (std::regex_match(fileName.string(), depthmapTemplate, std::regex_constants::match_any)) {
                depthmapFiles.push_back(file.string());
            }
        }

        // this gurantees files are loaded in the same order they where stored
        std::sort(grayscaleFiles.begin(), grayscaleFiles.end());
        std::sort(depthmapFiles.begin(),  depthmapFiles.end());

        vector<Mat> grayscaleID, depthmapID;
        for (const auto &file : grayscaleFiles){
            auto image = cv::imread(file, cv::IMREAD_GRAYSCALE);
            auto decoded = decode(image);
            grayscaleID.push_back(decoded);
        }
        for (const auto &file : depthmapFiles) {
            auto image = cv::imread(file, cv::IMREAD_GRAYSCALE);
            auto decoded = decode(image);
            depthmapID.push_back(decoded);
        }

        directoryMap.push_back(subdir.path().filename());

        grayscale.push_back(std::move(grayscaleID));
        depthmap.push_back(std::move(depthmapID));
    }
}


// ------------------------------------------------------------------------
// - Encoding and decoding functions to save and load covariance matrixes -
// ------------------------------------------------------------------------

uint32_t bitwiseFloatToUInt32_t(float x) {
    union { float f; uint32_t u; } converter;
    converter.f = x;
    return converter.u;
}

float bitwiseUInt32_tToFloat(uint32_t x) {
    union { float f; uint32_t u; } converter;
    converter.u = x;
    return converter.f;
}

cv::Mat encode(const cv::Mat& image)
{
    const auto HEIGHT = image.rows;
    const auto WIDTH = image.cols;
    cv::Mat encoded(HEIGHT, WIDTH * 4, CV_8U);

    for (auto i = 0; i < HEIGHT; ++i) {
        for (auto j = 0; j < WIDTH; ++j) {
            uint32_t value = bitwiseFloatToUInt32_t(image.at<float>(i, j));
            for (auto k = 0; k < 4; ++k) {
                encoded.at<uint8_t>(i, 4 * j + k) = value & 0XFF;
                value = value >> 8;
            }
        }
    }

    return encoded;
}

cv::Mat decode(const cv::Mat& image)
{
    const auto HEIGHT = image.rows;
    const auto WIDTH = image.cols / 4;
    auto decoded = cv::Mat(HEIGHT, WIDTH, CV_32FC1);

    for (auto i = 0; i < HEIGHT; ++i) {
        for (auto j = 0; j < WIDTH; ++j) {
            uint32_t value = 0;
            for (auto k = 0; k < 4; ++k) {
                uint32_t tmp = image.at<uint8_t>(i, 4 * j + k);
                value |= tmp << k * 8;
            }
            float floatVal = bitwiseUInt32_tToFloat(value);
            decoded.at<float>(i, j) = floatVal;
        }
    }

    return decoded;
}
}
