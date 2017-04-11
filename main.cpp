#include "backgroundsegmentation.h"
#include "imageloader.hpp"
#include "singletonsettings.h"
#include "yamlloader.h"
#include <iostream>
#include <opencv2/highgui.hpp>
//#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;

void testYaml()
{
    YamlLoader loader = YamlLoader();

    loader.setPath("camera_info.yaml");

    loader.read();

    SingletonSettings& settings = SingletonSettings::getInstance();
    cout << settings.getD() << endl
         << settings.getK() << endl
         << settings.getP() << endl
         << settings.getR() << endl
         << settings.getHeight() << endl
         << settings.getWidth() << endl;
}

void findThreshold()
{
    BackgroundSegmentation segmenter;

    string dirPath = "../RGBD_Face_dataset_training/";
    ImageLoader loader(dirPath, "014.*pcd"); // example: loads only .png files starting with 014

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    loader.get(cloud);

    segmenter.setImageDepth(cloud);

    cout << "Treshold found: " << segmenter.findTreshold() << endl;
}

void testImageLoader()
{
    string home = getenv("HOME");
    //    string dirPath = home + "/Pictures/RGBD_Face_dataset_testing/Test1";

    string dirPath = "../RGBD_Face_dataset_training/";
    ImageLoader loader(dirPath, "0_14.*png"); // example: loads only .png files

    while (loader.hasNext()) {
        Mat image;
        loader.get(image);
        imshow("image", image);

        waitKey(0);
    }
}

//void testCloudLoader()
//{
//    string home = getenv("HOME");
//    //    string dirPath = home + "/Pictures/RGBD_Face_dataset_testing/Test1";

//    string dirPath = "../RGBD_Face_dataset_training/";
//    ImageLoader loader(dirPath, "014.*pcd"); // example: loads only .png files starting with 014

//    while (loader.hasNext()) {
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
//        loader.get(cloud);
//        pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
//        viewer.showCloud(cloud);
//        while (!viewer.wasStopped()) {
//        }

//        waitKey(0);
//    }
//}

int main()
{
    cout << "Hello World!" << endl;
    testYaml();
    testImageLoader();
    findThreshold();
    //    testCloudLoader();
    return 0;
}
