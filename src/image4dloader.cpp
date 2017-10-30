#include "image4dloader.h"

#include <experimental/filesystem>

#include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "settings.h"

#include "image4d.h"

using cv::Mat;
using pcl::PointCloud;
using pcl::PointXYZ;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace fs = std::experimental::filesystem;

namespace face {

const string Image4DLoader::MATCH_ALL = ".*";
//const string Image4DLoader::NO_MATCH  = "\+";

Image4DLoader::Image4DLoader()
    : Image4DLoader(fs::current_path().string())
{
}

Image4DLoader::Image4DLoader(const string& dirPath, const string& fileNameRegEx)
    : currentPath(dirPath)
    , fileTemplate(fileNameRegEx)
{
    if (!loadFileNames(currentPath))
        cout << "Failed!" << endl;
}

bool Image4DLoader::hasNext() const
{
    return !imageFileNames.empty() && !cloudFileNames.empty();
}

bool Image4DLoader::get(Image4D& image4d)
{
    if (!hasNext())
        return false;

    const string &imageFile = imageFileNames.back();
    const string &cloudFile = cloudFileNames.back();

    Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) {
        cout << "Unable to load file " << imageFile << endl;
        return false;
    }

    PointCloud<PointXYZ> cloud;
    int result = pcl::io::loadPCDFile<PointXYZ>(cloudFile, cloud);
    if (result == -1) {
        cout << "Unable to load file " << cloudFile << endl;
        return false;
    }

    if (!cloud.isOrganized()) {
        std::cerr << "ERROR: loading unorganized point cloud!" << endl;
        return false;
    }

    Mat depthMap(cloud.height, cloud.width, CV_16SC1);
    for (uint x = 0; x < cloud.height; ++x) {
        for (uint y = 0; y < cloud.width; ++y) {
            depthMap.at<uint16_t>(x, y) = cloud.at(y, x).z * 10E2;
        }
    }

    image4d = Image4D(image, depthMap, Settings::getInstance().getK());

    imageFileNames.pop_back();
    cloudFileNames.pop_back();

    return true;
}

void Image4DLoader::getMultiThr(vector<Image4D>& image4DSequence, int begin, int end, std::mutex& mutex) const
{
    Mat K = Settings::getInstance().getK();

    for (int i = begin; i < end; ++i) {

        // no locks required since reading a const reference
        const string &imageFile = imageFileNames[i];
        const string &cloudFile = cloudFileNames[i];

        Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
        if (image.empty()) {
            cout << "Unable to load file " << imageFile << endl;
            return;
        }

        PointCloud<PointXYZ> cloud;
        int result = pcl::io::loadPCDFile<PointXYZ>(cloudFile, cloud);
        if (result == -1) {
            cout << "Unable to load file " << cloudFile << endl;
            return;
        }

        if (!cloud.isOrganized()) {
            std::cerr << "ERROR: loading unorganized point cloud!" << endl;
            return;
        }

        Mat depthMap(cloud.height, cloud.width, CV_16SC1);
        for (uint x = 0; x < cloud.height; ++x) {
            for (uint y = 0; y < cloud.width; ++y) {
                depthMap.at<uint16_t>(x, y) = cloud.at(y, x).z * 10E2;
            }
        }

        // lock needed to prevent concurrent writing
        std::lock_guard<std::mutex> lock(mutex);
        image4DSequence[i] = Image4D(image, depthMap, K);
        image4DSequence[i].name = imageFile.substr(0, imageFile.length() - 4);
    }
}

vector<Image4D> Image4DLoader::get()
{
    const auto SIZE = imageFileNames.size();
    vector<Image4D> image4DSequence(SIZE);

    // get number of concurrently executable threads
    const int numOfThreads = std::thread::hardware_concurrency();
    vector<std::thread> threads(numOfThreads);

    // equally split number of images to load
    int blockSize = SIZE / numOfThreads;
    if (blockSize < 1)
        blockSize = 1;

    std::mutex imageSeqMutex;
    for (int i = 0; i < numOfThreads && i < SIZE; ++i) {
        int begin = i * blockSize;
        int end = begin + blockSize;
        if (i == numOfThreads - 1)
            end += SIZE % numOfThreads;

        // start a thread executing getMultiThr function
        threads[i] = std::thread(&Image4DLoader::getMultiThr, this, std::ref(image4DSequence), begin, end, std::ref(imageSeqMutex));
    }

    // wait for threads to start
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // FIXME: this is not a safe way to do it

    // wait for threads to end (syncronization)
    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }

    imageFileNames.clear();
    cloudFileNames.clear();

    return image4DSequence;
}

void Image4DLoader::setFileNameRegEx(const string& fileNameRegEx)
{
    imageFileNames.clear();
    cloudFileNames.clear();
    fileTemplate = std::regex(fileNameRegEx);
    loadFileNames(currentPath);
}
/*
void Image4DLoader::clearFileNameRegEx()
{
    imageFileNames.clear();
    cloudFileNames.clear();
}
*/

void Image4DLoader::setCurrentPath(const string& dirPath)
{
    imageFileNames.clear();
    cloudFileNames.clear();
    currentPath = dirPath;
    loadFileNames(currentPath);
}

float Image4DLoader::getLeafSize() const
{
    return leafSize;
}

void Image4DLoader::setLeafSize(float value)
{
    leafSize = value;
}

bool Image4DLoader::loadFileNames(const string& dirPath)
{
    fs::path full_path = fs::system_complete(fs::path(dirPath));

    // check if exsists
    if (!fs::exists(full_path)) {
        std::cerr << "\nNot found: " << full_path.filename() << endl;
        return false;
    }

    // check if directory
    if (!fs::is_directory(full_path)) {
        std::cerr << "\n"
                  << full_path.filename() << " is not a directory" << endl;
        return false;
    }

    // iterate trough files and save
    fs::directory_iterator iter(full_path);
    for (auto& dir_entry : iter) {
        try {
            if (fs::is_regular_file(dir_entry.status())) {
                const fs::path& path = dir_entry.path();
                if (matchTemplate(path.stem().string())) {
                    if (path.extension().string().compare(".png") == 0)
                        imageFileNames.push_back(path.string());
                    else if (path.extension().string().compare(".pcd") == 0)
                        cloudFileNames.push_back(path.string());
                }
            }
        } catch (const std::exception& ex) {
            std::cerr << dir_entry.path().filename() << " " << ex.what() << endl;
            return false;
        }
    }

    std::sort(imageFileNames.begin(), imageFileNames.end());
    std::sort(cloudFileNames.begin(), cloudFileNames.end());

    return true;
}

bool Image4DLoader::matchTemplate(const string& fileName)
{
    return std::regex_match(fileName, fileTemplate, std::regex_constants::match_any);
}

} // face
