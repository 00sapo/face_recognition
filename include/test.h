#ifndef TEST_H
#define TEST_H

#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/visualization/area_picking_event.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "backgroundsegmentation.h"
#include "face.h"
#include "faceloader.h"
#include "singletonsettings.h"
#include "utils.h"

using namespace std;
using namespace cv;
//using namespace pcl;

namespace test {

void testSingletonSettings()
{
    cout << "SingletonSettings test..." << endl;
    SingletonSettings& settings = SingletonSettings::getInstance();
    cout << settings.getD() << endl
         << settings.getK() << endl
         << settings.getP() << endl
         << settings.getR() << endl
         << settings.getHeight() << endl
         << settings.getWidth() << endl;
    system("read -p 'Press [enter] to continue'");
}

void testFaceLoader()
{
    cout << "\n\nFace loader test..." << endl;
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

        viewPointCloud(face.depthMap);
    }
    system("read -p 'Press [enter] to continue'");
}

void testFindThreshold()
{
    cout << "\n\nFind threshold test..." << endl;
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

    BackgroundSegmentation segmenter;
    imshow("image", face.image);
    while (waitKey(0) != 'm') {
    }

    segmenter.removeBackground(face);

    //cout << "Treshold found: " << segmenter.findTreshold() << endl;
    imshow("image", face.image);
    while (waitKey(0) != 'm') {
    }
    viewPointCloud(face.depthMap);

    Mat depthMap = face.get3DImage(SingletonSettings::getInstance().getK());
    imshow("Depth Map", depthMap);
    waitKey(0);
    system("read -p 'Press [enter] to continue'");
}

void testGetDepthMap()
{
    cout << "\n\nGet depth map test..." << endl;
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "014.*"); // example: loads only .png files starting with 014

    Face face;

    //    loader.setDownscalingRatio(0.5);

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }

    cout << "Face loaded!" << endl;

    cv::Mat depthMap = face.get3DImage(SingletonSettings::getInstance().getK());

    imshow("Depth Map", depthMap);
    waitKey(0);
    system("read -p 'Press [enter] to continue'");
}

void testDetectFacePose()
{
    cout << "\n\nDetect face pose..." << endl;
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "000_.*"); // example: loads only .png files starting with 014

    Face face;

    //    loader.setDownscalingRatio(0.5);

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }

    cout << "Face loaded!" << endl;

    BackgroundSegmentation segmenter;
    cv::Rect detectedRegion;
    if (segmenter.detectForegroundFace(face, detectedRegion)) {
        cv::rectangle(face.image, detectedRegion, Scalar(255, 255, 255), 5);
        cv::imshow("Face detected", face.image);
        cv::waitKey(0);
    } else {
        std::cout << "No face detected!" << std::endl;
    }

    face.crop(detectedRegion);

    std::cout << "Removing background..." << std::endl;
    viewPointCloud(face.depthMap);
    imshow("Face", face.image);
    waitKey(0);
    segmenter.removeBackground(face);
    viewPointCloud(face.depthMap);
    imshow("Face", face.image);
    waitKey(0);
    std::cout << "Done!" << std::endl;

    std::cout << "Estimating face pose..." << std::endl;
    segmenter.estimateFacePose(face);
    system("read -p 'Press [enter] to continue'");
}

void testFaceDetection()
{
    cout << "\n\nDetect faces test..." << endl;
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "000_.*"); // example: loads only .png files starting with 014

    Face face;

    //    loader.setDownscalingRatio(0.5);

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }

    cout << "Face loaded!" << endl;

    BackgroundSegmentation segmenter;
    cv::Rect detectedRegion;
    if (segmenter.detectForegroundFace(face, detectedRegion)) {
        cv::rectangle(face.image, detectedRegion, Scalar(255, 255, 255), 5);

        imshow("image", face.image);
        waitKey(0);
    } else {
        std::cout << "No face detected!" << std::endl;
    }

    face.crop(detectedRegion);
    imshow("image", face.image);
    waitKey(0);

    viewPointCloud(face.depthMap);
    system("read -p 'Press [enter] to continue'");
}

void testKmeans()
{
    cout << "\n\nKmeans test..." << endl;
    vector<float> depth;

    depth.push_back(1.543);
    depth.push_back(1.563);
    depth.push_back(1.547);
    depth.push_back(1.743);
    depth.push_back(5.543);
    depth.push_back(5.673);
    depth.push_back(1.915);
    depth.push_back(1.543);
    depth.push_back(5.563);
    depth.push_back(4.743);
    depth.push_back(6.542);
    depth.push_back(1.246);

    cv::Mat centers(1, 2, CV_32F);
    std::vector<int> bestLabels;
    std::cout << "Clustering..." << std::endl;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
    std::cout << "Done!" << std::endl;

    std::cout << "Size: " << centers.size() << std::endl;
    //std::cout << "Cols: " << centers.cols << std::endl;

    std::cout << "Labels for points..." << std::endl;
    for (uint i = 0; i < bestLabels.size(); ++i) {
        std::cout << bestLabels.at(i) << std::endl;
    }
    std::cout << "Done!" << std::endl;

    system("read -p 'Press [enter] to continue'");
}
}

#endif // TEST_H
