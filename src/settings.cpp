#include "settings.h"

#include <iostream>

using cv::FileStorage;
using cv::Mat;
using std::cerr;
using std::cout;
using std::endl;

namespace face {

std::string Settings::cameraInfoPath = "camera_info.yaml";
bool Settings::cameraInfoPathHasChanged = false;

void Settings::setCameraInfoPath(const std::string& pathName)
{
    cameraInfoPath = pathName;
    cameraInfoPathHasChanged = true;
}

Settings& Settings::getInstance()
{
    static Settings instance;
    if (cameraInfoPathHasChanged) {
        instance.readCameraInfo();
        cameraInfoPathHasChanged = false;
    }
    return instance;
}

const Mat Settings::getK() { return K; }
const Mat Settings::getD() { return D; }
const Mat Settings::getP() { return P; }
const Mat Settings::getR() { return R; }
int Settings::getHeight() { return height; }
int Settings::getWidth() { return width; }

Settings::Settings()
{
    readCameraInfo();
}

bool Settings::readCameraInfo()
{
    // Opening file
    FileStorage fs(cameraInfoPath, FileStorage::READ); // FileStorage destructor is going to close the file.

    if (!fs.isOpened()) {
        cerr << "Failed to open " << cameraInfoPath << endl;
        return false;
    }

    // reading parameters
    fs["K"] >> K;
    fs["D"] >> D;
    fs["R"] >> R;
    fs["height"] >> height;
    fs["width"] >> width;
    fs["P"] >> P;

    return true;
}

std::string Settings::getPoseEstimatorPath() const
{
    return poseEstimatorPath;
}

std::string Settings::getFaceDetectorPath() const
{
    return faceDetectorPath;
}

} // face
