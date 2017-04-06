#include <iostream>
#include <opencv2/highgui.hpp>

#include "singletonsettings.h"
#include "yamlloader.h"
#include "imageloader.hpp"

using namespace std;
using namespace cv;

void testYaml()
{
    YamlLoader loader = YamlLoader();

    //    loader.setPath("../example.yaml");
    loader.setPath("camera_info.yaml");

    loader.read();

    SingletonSettings* settings = SingletonSettings::getInstance();
    cout << settings->getD() << endl
         << settings->getK() << endl
         << settings->getP() << endl
         << settings->getR() << endl
         << settings->getHeight() << endl
         << settings->getWidth() << endl;
}

void testImageLoader() {
    string home = getenv("HOME");
    string dirPath = home + "/Pictures/RGBD_Face_dataset_testing/Test1";
    ImageLoader loader(dirPath, ".*png");       // example: loads only .png files
    //ImageLoader loader(dirPath, "014.*png");  // example: loads only .png files starting with 014
    while(loader.hasNext()) {
        Mat image;
        loader.get(image);
        imshow("image", image);
        waitKey(0);
    }
}

int main()
{
    cout << "Hello World!" << endl;
    testYaml();
    testImageLoader();
    return 0;
}
