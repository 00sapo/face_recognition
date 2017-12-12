#include "datasetcov.h"

#include <iostream>

#include <opencv2/highgui.hpp>

using std::vector;
using std::string;
using cv::Mat;

namespace face {

DatasetCov::DatasetCov() : consistent(true) { }

DatasetCov::DatasetCov(vector<vector<Mat>> grayscale, vector<vector<Mat>> depthmap)
    : grayscale(grayscale), depthmap(depthmap)
{
    consistent = checkConsistency();
}

bool DatasetCov::isConsistent() const
{
    return consistent;
}

bool DatasetCov::checkConsistency() const
{
    if (grayscale.size() != depthmap.size())
        return false;

    for (auto i = 0; i < grayscale.size(); ++i) {
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
            std::cout << rootDir << " already exists. " <<
                "Do you want to delete its content and write the dataset to this folder? [y|n]" << std::endl;
            char answer;
            std::cin >> answer;
            if (answer == 'y') {
                fs::remove_all(rootDir);
                saved = save(rootDir);
                done = true;
            }
            else if (answer == 'n') {
                std::cout << "Dataset not saved." << std::endl;
                done = true;
            }
        }
    }
    else {
        saved = save(rootDir);
    }
    return saved;
}


bool DatasetCov::save(const fs::path& path)
{
    try {   // try creating dataset root directory
        if (!fs::create_directory(path))
            return false;
    } catch (const fs::filesystem_error& fsex) {
        std::cerr << fsex.what() << std::endl;
        return false;
    }

    bool success = true;
    for (auto i = 0; i < grayscale.size(); ++i) {   // for each identity
        const auto& grayscaleID = grayscale[i];
        const auto& depthmapID  = depthmap[i];
        assert (grayscaleID.size() == depthmapID.size() && "Identities size mismatch!");

        auto idPath = path / std::to_string(i);

        try {   // try creating identity's directory
            if (!fs::create_directory(idPath))
                return false;
        } catch (const fs::filesystem_error& fsex) {
            std::cerr << fsex.what() << std::endl;
            return false;
        }

        for (auto j = 0; j < grayscaleID.size(); ++j) { // save Mats
            if (grayscaleID[j].empty() || depthmapID[j].empty())
                std::cout << "Warning! Trying to save empty image!" << std::endl;

            auto grayscalePath = idPath / ("grayscale_" + std::to_string(j) + ".png");
            auto depthmapPath  = idPath / ("depthmap_" + std::to_string(j) + ".png");

            success &= cv::imwrite(grayscalePath.string(), grayscaleID[j]);
            success &= cv::imwrite(depthmapPath.string(),  depthmapID[j]);
        }
    }

    return success;
}


//static DatasetCov DatasetCov::load(const std::string& path);





}
