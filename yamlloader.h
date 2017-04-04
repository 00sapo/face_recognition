#ifndef YAMLLOADER_H
#define YAMLLOADER_H
#include "iostream"
#include "opencv2/core.hpp"
#include "singletonsettings.h"
#include "string"

using namespace std;
using namespace cv;

class YamlLoader {
public:
    YamlLoader();

    /**
     * @brief read read camera parameters from Yaml file
     * @param path path of the yaml file
     * @return false if errors occured, true otherwise
     */

    bool read(string path);

    /**
     * @brief read read camera parameters from Yaml file using url setted
     * @return false if errors occured, true otherwise
     */
    bool read();
    void setPath(string path);

private:
    string path;
};

#endif // YAMLLOADER_H
