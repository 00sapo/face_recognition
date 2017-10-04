#include "settings.h"

#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using cv::Mat;
using cv::FileStorage;

namespace face {


std::string Settings::path = "camera_info.yaml";
bool Settings::pathHasChanged = false;

void Settings::setPath(const std::string &pathName)
{
    path = pathName;
    pathHasChanged = true;
}

Settings& Settings::getInstance()
{
    static Settings instance;
    if(pathHasChanged) {
        instance.read();
        pathHasChanged = false;
    }
    return instance;
}

const Mat Settings::getK() { return K; }
const Mat Settings::getD() { return D; }
const Mat Settings::getP() { return P; }
const Mat Settings::getR() { return R; }
int Settings::getHeight()  { return height; }
int Settings::getWidth()   { return width;  }

Settings::Settings()
{
    read();
}

bool Settings::read() {
    // Opening file
    cout << "\nReading: " << endl;
    FileStorage fs(path, FileStorage::READ);    // FileStorage destructor is going to close the file.

    if (!fs.isOpened()) {
        cerr << "Failed to open " << path << endl;
        return false;
    }

    // reading parameters
    fs["K"]      >> K;
    fs["D"]      >> D;
    fs["R"]      >> R;
    fs["height"] >> height;
    fs["width"]  >> width;
    fs["P"]      >> P;

    return true;
}

} // face
