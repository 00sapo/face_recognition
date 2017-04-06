#include "yamlloader.h"

#include <iostream>
#include <opencv2/core.hpp>

#include "singletonsettings.h"

using namespace cv;

YamlLoader::YamlLoader()
{
}

bool YamlLoader::read(string path)
{
    setPath(path);
    return read();
}

bool YamlLoader::read()
{
    /* Opening file */
    cout << endl
         << "Reading: " << endl;
    FileStorage fs;
    fs.open(path, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Failed to open " << path << endl;
        return false;
    }

    /* reading parameters */
    Mat K, D, R, P;
    int height, width;
    fs["K"] >> K; // Read cv::Mat
    fs["D"] >> D;
    fs["R"] >> R;
    fs["height"] >> height;
    fs["width"] >> width;
    fs["P"] >> P;

    /* saving parameters to SingletonSettings */
    SingletonSettings* settings = SingletonSettings::getInstance();
    settings->setD(D);
    settings->setK(K);
    settings->setP(P);
    settings->setR(R);
    settings->setHeight(height);
    settings->setWidth(width);

    return true;
}

void YamlLoader::setPath(string path)
{
    this->path = path;
}
