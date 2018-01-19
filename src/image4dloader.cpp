#include "image4dloader.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "image4d.h"

using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace face {

Mat loadDepthImageCompressed(const string& fname);
std::vector<cv::Point3f> cvtDepthMapTo3DVec(const cv::Mat& depthMap, const cv::Mat& intrinsicMatrix);

//----------------------------------------------
//--------------- CalibParams-------------------
//----------------------------------------------

Image4DLoader::CalibParams::CalibParams()
    : depthCameraMatrix(3, 3, CV_32FC1), rgbCameraMatrix(3, 3, CV_32FC1), rgbRotationMatrix(3, 3, CV_32FC1)
{
}

bool Image4DLoader::CalibParams::load(const fs::path& dir)
{
    auto depthCalibFile = dir / "depth.cal";
    auto rgbCalibFile = dir / "rgb.cal";

    std::ifstream depthFile(depthCalibFile);
    std::ifstream rgbFile(rgbCalibFile);
    if (!depthFile.is_open() || !rgbFile.is_open())
        return false;

    depthFile >> depthCameraMatrix.at<float>(0, 0);
    depthFile >> depthCameraMatrix.at<float>(0, 1);
    depthFile >> depthCameraMatrix.at<float>(0, 2);
    depthFile >> depthCameraMatrix.at<float>(1, 0);
    depthFile >> depthCameraMatrix.at<float>(1, 1);
    depthFile >> depthCameraMatrix.at<float>(1, 2);
    depthFile >> depthCameraMatrix.at<float>(2, 0);
    depthFile >> depthCameraMatrix.at<float>(2, 1);
    depthFile >> depthCameraMatrix.at<float>(2, 2);

    rgbFile >> rgbCameraMatrix.at<float>(0, 0);
    rgbFile >> rgbCameraMatrix.at<float>(0, 1);
    rgbFile >> rgbCameraMatrix.at<float>(0, 2);
    rgbFile >> rgbCameraMatrix.at<float>(1, 0);
    rgbFile >> rgbCameraMatrix.at<float>(1, 1);
    rgbFile >> rgbCameraMatrix.at<float>(1, 2);
    rgbFile >> rgbCameraMatrix.at<float>(2, 0);
    rgbFile >> rgbCameraMatrix.at<float>(2, 1);
    rgbFile >> rgbCameraMatrix.at<float>(2, 2);

    // skip useless params
    for (auto i = 0; i < 4; ++i) {
        float useless;
        rgbFile >> useless;
    }

    rgbFile >> rgbRotationMatrix.at<float>(0, 0);
    rgbFile >> rgbRotationMatrix.at<float>(0, 1);
    rgbFile >> rgbRotationMatrix.at<float>(0, 2);
    rgbFile >> rgbRotationMatrix.at<float>(1, 0);
    rgbFile >> rgbRotationMatrix.at<float>(1, 1);
    rgbFile >> rgbRotationMatrix.at<float>(1, 2);
    rgbFile >> rgbRotationMatrix.at<float>(2, 0);
    rgbFile >> rgbRotationMatrix.at<float>(2, 1);
    rgbFile >> rgbRotationMatrix.at<float>(2, 2);

    rgbFile >> rgbTranslationVector[0];
    rgbFile >> rgbTranslationVector[1];
    rgbFile >> rgbTranslationVector[2];

    return true;
}

// ---------------------------------------------------
// ---------------- Image4DLoader --------------------
// ---------------------------------------------------

const string Image4DLoader::MATCH_ALL = ".*";

Image4DLoader::Image4DLoader()
    : Image4DLoader(fs::current_path().string())
{
}

Image4DLoader::Image4DLoader(const string& dirPath, const string& fileNameRegEx)
    : currentPath(dirPath)
    , fileTemplate(fileNameRegEx)
{
    loadMetadata(currentPath);
}

bool Image4DLoader::hasNext() const
{
    return !imageFileNames.empty() && !depthFileNames.empty();
}

bool Image4DLoader::get(Image4D& image4D)
{
    if (!hasNext())
        return false;

    const auto& imageFile = imageFileNames.back();
    const auto& depthFile = depthFileNames.back();

    if (!get(imageFile, depthFile, image4D))
        return false;

    imageFileNames.pop_back();
    depthFileNames.pop_back();

    return true;
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

        // start a thread executing threadGet function
        threads[i] = std::thread(&Image4DLoader::threadGet, this, std::ref(image4DSequence), begin, end, std::ref(imageSeqMutex));
    }

    // wait for threads to end (syncronization)
    for (auto& thread : threads) {
        thread.join();
    }

    imageFileNames.clear();
    depthFileNames.clear();

    return image4DSequence;
}

void Image4DLoader::threadGet(vector<Image4D>& image4DSequence, int begin, int end, std::mutex& mutex) const
{
    for (int i = begin; i < end; ++i) {
        // no locks required since reading a const reference
        const auto& imageFile = imageFileNames[i];
        const auto& depthFile = depthFileNames[i];

        Image4D image4D;
        get(imageFile, depthFile, image4D);

        // lock needed to prevent concurrent writing
        std::lock_guard<std::mutex> lock(mutex);
        image4DSequence[i] = image4D;
        image4DSequence[i].name = (imageFile.parent_path() / imageFile.stem()).string();
    }
}

bool Image4DLoader::get(const fs::path& imageFile, const fs::path& depthFile, Image4D& image4D) const
{
    // load grayscale image
    auto grayscale = cv::imread(imageFile.string(), CV_LOAD_IMAGE_GRAYSCALE);
    if (grayscale.empty()) {
        std::cerr << "Unable to load file " << imageFile << std::endl;
        return false;
    }

    // load depth image
    auto depthMap = loadDepthImageCompressed(depthFile.string());
    if (depthMap.empty()) {
        std::cerr << "Unable to load file " << depthFile << std::endl;
        return false;
    }

    // --- align depth image with grayscale image ---

    auto points3D = cvtDepthMapTo3DVec(depthMap, calibParams.depthCameraMatrix); // compute 3D points from depth image

    // project 3D points to grayscale image plane
    Mat rvec, projPoints;
    cv::Rodrigues(calibParams.rgbRotationMatrix,rvec);
    cv::projectPoints(points3D, rvec, calibParams.rgbTranslationVector, calibParams.rgbCameraMatrix, std::vector<float>(), projPoints);

    depthMap = Mat::zeros(depthMap.rows, depthMap.cols, depthMap.type()); // clear old depth map

    // fill depthMap with projected points
    for (auto k = 0; k < projPoints.rows; ++k) {
        const auto &point = projPoints.at<cv::Point2f>(k,1);
        if (point.x > 0 && point.x < depthMap.cols && point.y > 0 && point.y < depthMap.rows)
            depthMap.at<uint16_t>(point.y, point.x) = points3D[k].z;
    }

    image4D = Image4D(grayscale, depthMap, calibParams.rgbCameraMatrix);

    return true;
}

void Image4DLoader::setFileNameRegEx(const string& fileNameRegEx)
{
    imageFileNames.clear();
    depthFileNames.clear();
    fileTemplate = std::regex(fileNameRegEx);
    loadMetadata(currentPath);
}

void Image4DLoader::setCurrentPath(const string& dirPath)
{
    imageFileNames.clear();
    depthFileNames.clear();
    currentPath = fs::path(dirPath);
    loadMetadata(currentPath);
}

bool Image4DLoader::loadMetadata(const fs::path& dirPath)
{
    fs::path full_path = fs::system_complete(dirPath);

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
                const auto& path = dir_entry.path();
                if (matchTemplate(path.stem().string())) {
                    if (path.extension().string().compare(".png") == 0)
                        imageFileNames.push_back(path);
                    else if (path.extension().string().compare(".bin") == 0)
                        depthFileNames.push_back(path);
                }
            }
        } catch (const std::exception& ex) {
            std::cerr << dir_entry.path().filename() << " " << ex.what() << endl;
            return false;
        }
    }

    std::sort(imageFileNames.begin(), imageFileNames.end());
    std::sort(depthFileNames.begin(), depthFileNames.end());

    if (!calibParams.load(full_path))
        return false;

    return true;
}

bool Image4DLoader::matchTemplate(const string& fileName)
{
    return std::regex_match(fileName, fileTemplate, std::regex_constants::match_any);
}

//-------------------------------------------------
//---------- Free functions -----------------------
//-------------------------------------------------

Mat loadDepthImageCompressed(const string& fname)
{
    //now read the depth image
    FILE* pFile = fopen(fname.c_str(), "rb");
    if (!pFile) {
        std::cerr << "could not open file " << fname << std::endl;
        return Mat();
    }

    int im_width = 0;
    int im_height = 0;
    bool success = true;

    success &= (fread(&im_width, sizeof(int), 1, pFile) == 1); // read width of depthmap
    success &= (fread(&im_height, sizeof(int), 1, pFile) == 1); // read height of depthmap

    Mat depth(im_height, im_width, CV_16SC1);
    auto depth_img = (uint16_t*)depth.data;

    int numempty, numfull;
    int p = 0;
    while (p < im_width * im_height) {

        success &= (fread(&numempty, sizeof(int), 1, pFile) == 1);

        for (int i = 0; i < numempty; i++)
            depth_img[p + i] = 0;

        success &= (fread(&numfull, sizeof(int), 1, pFile) == 1);
        success &= (fread(&depth_img[p + numempty], sizeof(int16_t), numfull, pFile) == (unsigned int)numfull);
        p += numempty + numfull;
    }

    fclose(pFile);

    if (success)
        return depth;

    return Mat();
}

std::vector<cv::Point3f> cvtDepthMapTo3DVec(const cv::Mat& depthMap, const cv::Mat& intrinsicMatrix)
{
    auto fx = intrinsicMatrix.at<float>(0,0);
    auto fy = intrinsicMatrix.at<float>(1,1);
    auto cx = intrinsicMatrix.at<float>(0,2);
    auto cy = intrinsicMatrix.at<float>(1,2);

    const auto HEIGHT = depthMap.rows;
    const auto WIDTH  = depthMap.cols;
    std::vector<cv::Point3f> points3D;

    for (uint i = 0; i < HEIGHT; ++i) {
        for (uint j = 0; j < WIDTH; ++j) {
            auto d = static_cast<float>(depthMap.at<uint16_t>(i,j));
            auto x = d * (float(j) - cx)/fx;
            auto y = d * (float(i) - cy)/fy;
            auto z = d;
            points3D.emplace_back(x,y,z);
        }
    }

    return points3D;
}

} // namespace face
