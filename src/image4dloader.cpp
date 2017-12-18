#include "image4dloader.h"

#include <thread>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <experimental/filesystem>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "settings.h"

#include "image4d.h"

using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace fs = std::experimental::filesystem;

namespace face {

Mat loadDepthImageCompressed( const string& fname ){

    //now read the depth image
    FILE* pFile = fopen(fname.c_str(), "rb");
    if(!pFile){
        std::cerr << "could not open file " << fname << std::endl;
        return Mat();
    }

    int im_width = 0;
    int im_height = 0;
    bool success = true;

    success &= ( fread(&im_width,sizeof(int),1,pFile) == 1 ); // read width of depthmap
    success &= ( fread(&im_height,sizeof(int),1,pFile) == 1 ); // read height of depthmap

    //int16_t* depth_img = new int16_t[im_width*im_height];
    Mat depth(im_height, im_width, CV_16SC1);
    auto depth_img = (uint16_t*)depth.data;

    int numempty;
    int numfull;
    int p = 0;
    while(p < im_width*im_height ){

        success &= ( fread( &numempty,sizeof(int),1,pFile) == 1 );

        for(int i = 0; i < numempty; i++)
            depth_img[ p + i ] = 0;

        success &= ( fread( &numfull,sizeof(int), 1, pFile) == 1 );
        success &= ( fread( &depth_img[ p + numempty ], sizeof(int16_t), numfull, pFile) == (unsigned int) numfull );
        p += numempty+numfull;

    }

    fclose(pFile);

    if(success)
        return depth;

    return Mat();
}

Mat loadCalibrationData(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        return Mat();

    Mat intrinsics = Mat::zeros(3,3,CV_32FC1);
    file >> intrinsics.at<float>(0,0);
    file >> intrinsics.at<float>(0,1);
    file >> intrinsics.at<float>(0,2);
    file >> intrinsics.at<float>(1,0);
    file >> intrinsics.at<float>(1,1);
    file >> intrinsics.at<float>(1,2);
    file >> intrinsics.at<float>(2,0);
    file >> intrinsics.at<float>(2,1);
    file >> intrinsics.at<float>(2,2);

    return intrinsics;
}



// ---------------------------------------------------
// ---------------- Image4DLoader --------------------
// ---------------------------------------------------

const string Image4DLoader::MATCH_ALL = ".*";
//const string Image4DLoader::NO_MATCH  = "\+";

Image4DLoader::Image4DLoader()
    : Image4DLoader(fs::current_path().string())
{
}

Image4DLoader::Image4DLoader(const string& dirPath, const string& fileNameRegEx)
    : currentPath(dirPath), fileTemplate(fileNameRegEx)
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

    auto image = cv::imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Unable to load file " << imageFile << std::endl;
        return false;
    }

    auto depthMap = loadDepthImageCompressed(cloudFile);
    if (depthMap.empty()) {
        std::cerr << "Unable to load file " << cloudFile << std::endl;
        return false;
    }

    auto calibFile = currentPath + "/" + "depth.cal";
    auto intrinsics = loadCalibrationData(calibFile);
    if (intrinsics.empty()) {
        std::cerr << "Unable to load file " << calibFile << std::endl;
        return false;
    }

    image4d = Image4D(image, depthMap, intrinsics);

    imageFileNames.pop_back();
    cloudFileNames.pop_back();

    return true;
}

void Image4DLoader::getMultiThr(vector<Image4D>& image4DSequence, int begin, int end, std::mutex& mutex) const
{
    //Mat K = Settings::getInstance().getK();

    for (int i = begin; i < end; ++i) {

        // no locks required since reading a const reference
        const string &imageFile = imageFileNames[i];
        const string &cloudFile = cloudFileNames[i];

        Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Unable to load file " << imageFile << std::endl;
            return;
        }

        auto depthMap = loadDepthImageCompressed(cloudFile);
        if (depthMap.empty()) {
            std::cerr << "Unable to load file " << cloudFile << std::endl;
            return;
        }
        auto calibFile = currentPath + "/" + "depth.cal";
        auto intrinsics = loadCalibrationData(calibFile);
        if (intrinsics.empty()) {
            std::cerr << "Unable to load file " << calibFile << std::endl;
            return;
        }

        // lock needed to prevent concurrent writing
        std::lock_guard<std::mutex> lock(mutex);
        image4DSequence[i] = Image4D(image, depthMap, intrinsics);
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
        std::cerr << "\nNot found: " << full_path.filename() << std::endl;
        return false;
    }

    // check if directory
    if (!fs::is_directory(full_path)) {
        std::cerr << "\n"
                  << full_path.filename() << " is not a directory" << std::endl;
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
                    else if (path.extension().string().compare(".bin") == 0)
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

} // namespace face
