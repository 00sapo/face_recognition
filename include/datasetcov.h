#ifndef FACE_COVARIANCEDATASET_H
#define FACE_COVARIANCEDATASET_H

#include <vector>
#include <experimental/filesystem>

#include <opencv2/core.hpp>

namespace fs = std::experimental::filesystem;

namespace face {

class DatasetCov {

    bool consistent;

    bool save(const fs::path& path);

public:

    std::vector<std::vector<cv::Mat>> grayscale;
    std::vector<std::vector<cv::Mat>> depthmap;

    DatasetCov();
    DatasetCov(std::vector<std::vector<cv::Mat>> grayscale, std::vector<std::vector<cv::Mat>> depthmap);

    bool isConsistent() const;
    bool checkConsistency() const;

    bool save(const std::string& path);
    //static DatasetCov load(const std::string& path);

};

}

#endif // FACE_COVARIANCEDATASET_H
