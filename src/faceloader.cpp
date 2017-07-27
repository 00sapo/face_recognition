#include "faceloader.h"

//#include <iostream>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "singletonsettings.h"

#include "image4d.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Mat;
using pcl::PointCloud;
using pcl::PointXYZ;

namespace fs = boost::filesystem;



const string FaceLoader::MATCH_ALL = ".*";


FaceLoader::FaceLoader()
{
    currentPath = fs::current_path().string();
    fileTemplate = std::regex(".*(png|jpg|bmp)");
    imageFileNames = vector<string>();
    cloudFileNames = vector<string>();
}

FaceLoader::FaceLoader(const string& dirPath, const string& fileNameRegEx)
{
    imageFileNames = vector<string>();
    cloudFileNames = vector<string>();
    currentPath = dirPath;
    fileTemplate = std::regex(fileNameRegEx);

    cout << "FaceLoader constructor: loading file names..." << endl;
    if (!loadFileNames(currentPath))
        cout << "Failed!" << endl;
}

bool FaceLoader::hasNext() const
{
    return !imageFileNames.empty() && !cloudFileNames.empty();
}

bool FaceLoader::get(Image4D& face)
{

    if (!hasNext())
        return false;

    string& imageFile = imageFileNames.back();
    string& cloudFile = cloudFileNames.back();

    Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) {
        cout << "Unable to load file " << imageFile << endl;
        return false;
    }

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    int result = pcl::io::loadPCDFile<PointXYZ>(cloudFile, *cloud);
    if (result == -1) {
        cout << "Unable to load file " << cloudFile << endl;
        return false;
    }

    if(!cloud->isOrganized()) {
        std::cerr << "ERROR: loading unorganized point cloud!" << endl;
        return false;
    }

    Mat depthMap(cloud->height, cloud->width, CV_16SC1);
    for (uint x = 0; x < cloud->height; ++x) {
        for (uint y = 0; y < cloud->width; ++y) {
            depthMap.at<uint16_t>(x,y) = cloud->at(y,x).z * 10E2;
        }
    }

    face = Image4D(image, depthMap, SingletonSettings::getInstance().getK());

    imageFileNames.pop_back();
    cloudFileNames.pop_back();

    return true;
}

bool FaceLoader::get(vector<Image4D>& faceSequence)
{

    faceSequence.clear();
    faceSequence.reserve(imageFileNames.size());
    while (hasNext()) {
        faceSequence.emplace_back();
        if (!get(faceSequence.back()))
            return false;
    }

    return true;
}

void FaceLoader::setFileNameRegEx(const string& fileNameRegEx)
{
    fileTemplate = std::regex(fileNameRegEx);
    loadFileNames(currentPath);
}

void FaceLoader::setCurrentPath(const string& dirPath)
{
    imageFileNames.clear();
    cloudFileNames.clear();
    currentPath = dirPath;
    loadFileNames(currentPath);
}

float FaceLoader::getLeafSize() const
{
    return leafSize;
}

void FaceLoader::setLeafSize(float value)
{
    leafSize = value;
}

bool FaceLoader::loadFileNames(const string& dirPath)
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

bool FaceLoader::matchTemplate(const string& fileName)
{
    return std::regex_match(fileName, fileTemplate, std::regex_constants::match_any);
}
