#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/visualization/area_picking_event.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "backgroundsegmentation.h"
#include "face.h"
#include "faceloader.h"
#include "singletonsettings.h"

using namespace std;
using namespace cv;
using namespace pcl;

void testSingletonSettings()
{
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
    if (!loader.get(faceSequence)) {
        cout << "Error loading face!" << endl;
        return;
    }
    cout << "\n\nFaces loaded!" << endl;

    namedWindow("image", WINDOW_NORMAL);
    for (const auto& face : faceSequence) {
        imshow("image", face.image);
        while (waitKey(0) != 'm') {
        }

        viewPointCloud(face.cloud);
    }
}

void testFindThreshold()
{

    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "014.*"); // example: loads only .png files starting with 014

    Face face;

    //    loader.setDownscalingRatio(0.5);

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }

    cout << "Face loaded!" << endl;

    cout << "\nFiltering background..." << endl;

    BackgroundSegmentation segmenter(face);
    imshow("image", segmenter.getFace().image);
    while (waitKey(0) != 'm') {
    }

    segmenter.filterBackground();

    //cout << "Treshold found: " << segmenter.findTreshold() << endl;
    imshow("image", segmenter.getFace().image);
    while (waitKey(0) != 'm') {
    }
    viewPointCloud(segmenter.getFace().cloud);
}

void testGetDepthMap()
{
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "014.*"); // example: loads only .png files starting with 014

    Face face;

    //    loader.setDownscalingRatio(0.5);

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }

    cout << "Face loaded!" << endl;

    cv::Mat depthMap = face.getDepthMap();

    imshow("Depth Map", depthMap);
    waitKey(0);
}

int main()
{

    cout << "SingletonSettings test..." << endl;
    testSingletonSettings();

    cout << "\n\nFace loader test..." << endl;
    testFaceLoader();

    cout << "\n\nFind threshold test..." << endl;
    testFindThreshold();

    cout << "\n\nGet depth map test..." << endl;
    //testGetDepthMap();

    cout << "\n\nTests finished!" << endl;

    return 0;
}
