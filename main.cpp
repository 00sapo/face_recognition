#include <iostream>

#include <opencv2/highgui.hpp>
//#include <pcl/visualization/cloud_viewer.h>

//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>

#include "backgroundsegmentation.h"
#include "faceloader.h"
#include "singletonsettings.h"
#include "yamlloader.h"
#include "face.h"


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

void testFaceLoader()
{
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "014.*");
    vector<Face> faceSequence(0);
    if(!loader.get(faceSequence)) {
        cout << "Error loading face!" << endl;
        return;
    }
    cout << "\n\nFaces loaded!" << endl;

    namedWindow( "image", WINDOW_NORMAL );
    for (const auto& face : faceSequence) {
        imshow("image", face.image);
        waitKey(1000);


        //visualization::PCLVisualizer viewer("PCL Viewer");
        //viewer.setBackgroundColor  (0.0, 0.0, 0.5);
        //viewer.addCoordinateSystem (0.1);
        //viewer.initCameraParameters();
        ////visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
        //viewer.addPointCloud<PointXYZRGB> (cloud, "input_cloud");
        //
        //while (!viewer.wasStopped()) {
        //    viewer.spin();
        //}

    }
    destroyWindow("image");
}

void testFindThreshold()
{
    BackgroundSegmentation segmenter;

    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "014.*"); // example: loads only .png files starting with 014

    Face face;

    if(!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }


    cout << "Face loaded!" << endl;

    cout << "\nFiltering background..." << endl;

    segmenter.filterBackground(face);

    //cout << "Treshold found: " << segmenter.findTreshold() << endl;
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

    cout << "Yaml test..." << endl;
    testYaml();

    cout << "\n\nFace loader test..."  << endl;
    testFaceLoader();

    cout << "\n\nFind threshold test..." << endl;
    testFindThreshold();

    cout << "\n\nTests finished!" << endl;

    //    testCloudLoader();
    return 0;
}
