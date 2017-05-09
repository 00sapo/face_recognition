#include "singletonsettings.h"

#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using cv::Mat;
using cv::FileStorage;

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

/*
void SingletonSettings::setK(Mat K)
{
    this->K = K;
}

void SingletonSettings::setD(Mat D)
{
    this->D = D;
}

void SingletonSettings::setP(Mat P)
{
    this->P = P;
}

void SingletonSettings::setR(Mat R)
{
    this->R = R;
}

void SingletonSettings::setHeight(int height)
{
    this->height = height;
}

void SingletonSettings::setWidth(int width)
{
    this->width = width;
}
*/

SingletonSettings::SingletonSettings()
{
    read();
}

bool SingletonSettings::read() {
    /* Opening file */
    cout << "\nReading: " << endl;
    FileStorage fs;
    fs.open(path, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << path << endl;
        return false;
    }

    /* reading parameters */
    //Mat K, D, R, P;
    //int height, width;
    fs["K"]      >> K; // Read cv::Mat
    fs["D"]      >> D;
    fs["R"]      >> R;
    fs["height"] >> height;
    fs["width"]  >> width;
    fs["P"]      >> P;

    /* saving parameters to SingletonSettings */
    //SingletonSettings& settings = SingletonSettings::getInstance();
    //settings.setD(D);
    //settings.setK(K);
    //settings.setP(P);
    //settings.setR(R);
    //settings.setHeight(height);
    //settings.setWidth(width);

    return true;
}
