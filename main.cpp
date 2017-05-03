#include <iostream>
#include <vector>

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

    Mat depthMap = segmenter.getFace().get3DImage();
    imshow("Depth Map", depthMap);
    waitKey(0);
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

    cv::Mat depthMap = face.get3DImage();

    imshow("Depth Map", depthMap);
    waitKey(0);
}

void testDetectFacePose()
{

    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "000_.*"); // example: loads only .png files starting with 014

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

    segmenter.cropFace();
    waitKey(0);
}

void testFaceDetection() {
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "000_.*"); // example: loads only .png files starting with 014

    Face face;

    //    loader.setDownscalingRatio(0.5);

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }

    cout << "Face loaded!" << endl;

    BackgroundSegmentation segmenter(face);
    std::vector<cv::Rect> faces;
    if(segmenter.detectFaces(faces)) {
        for(const auto& rect : faces) {
            cv::rectangle(face.image, rect, Scalar(255,255,255), 5);
        }
        imshow("image", face.image);
        waitKey(0);
    }
    else {
        std::cout << "No face detected!" << std::endl;
    }
}

int main()
{

    cout << "SingletonSettings test..." << endl;
    //testSingletonSettings();

    cout << "\n\nFace loader test..." << endl;
    //testFaceLoader();

    cout << "\n\nFind threshold test..." << endl;
    //testFindThreshold();

    cout << "\n\nGet depth map test..." << endl;
    //testGetDepthMap();

    cout << "\n\nDetect face pose..." << endl;
    //testDetectFacePose();

    cout << "\n\nDetect faces test..." << endl;
    testFaceDetection();

    cout << "\n\nTests finished!" << endl;

    return 0;
}
