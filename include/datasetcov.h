#ifndef FACE_COVARIANCEDATASET_H
#define FACE_COVARIANCEDATASET_H

#include <experimental/filesystem>
#include <vector>

#include <opencv2/core.hpp>

namespace fs = std::experimental::filesystem;

namespace face {

class DatasetCov {

    bool consistent;

    bool save(const fs::path& path);

public:
    std::vector<std::vector<cv::Mat>> grayscale; // stores a vector of grayscale covariance matrixes for each identity
    std::vector<std::vector<cv::Mat>> depthmap; // stores a vector of depthmap  covariance matrixes for each identity

    DatasetCov();
    DatasetCov(std::vector<std::vector<cv::Mat>> grayscale, std::vector<std::vector<cv::Mat>> depthmap, std::vector<std::string> directoryMap);

    std::string getDirectory(int id) const;

    bool empty() const;

    /**
     * @brief clear calls clear on grayscale, depthmap and direcotryMap
     */
    void clear();

    /**
     * @brief isConsistent returns the last consistency state computed by checkConsistency()
     */
    bool isConsistent() const;

    /**
     * @brief Checks if grayscale and depthmap have the same size() and if the
     *        vector at position i in grayscale has the same size of the vector
     *        at position i in depthmap (for i in [0, grayscale.size()]).
     *        Then stores the result in DatasetCov::consistent.
     */
    bool checkConsistency() const;

    bool save(const std::string& path);

    void load(const std::string& path); //, std::vector<std::string> &idMap);

private:
    std::vector<std::string> directoryMap;
};
}

#endif // FACE_COVARIANCEDATASET_H
