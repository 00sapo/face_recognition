#include "singletonsettings.h"

#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using cv::Mat;
using cv::FileStorage;

namespace face {


std::string SingletonSettings::path = "camera_info.yaml";
bool SingletonSettings::pathHasChanged = false;

void SingletonSettings::setPath(const std::string &pathName)
{
    path = pathName;
    pathHasChanged = true;
}

SingletonSettings& SingletonSettings::getInstance()
{
    static SingletonSettings instance;
    if(pathHasChanged) {
        instance.read();
        pathHasChanged = false;
    }
    return instance;
}

const Mat SingletonSettings::getK() { return K; }
const Mat SingletonSettings::getD() { return D; }
const Mat SingletonSettings::getP() { return P; }
const Mat SingletonSettings::getR() { return R; }
int SingletonSettings::getHeight()  { return height; }
int SingletonSettings::getWidth()   { return width;  }

SingletonSettings::SingletonSettings()
{
    read();
}

bool SingletonSettings::read() {
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
