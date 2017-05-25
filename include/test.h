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

using namespace std;
using namespace cv;
using namespace pcl;

namespace test {

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

    segmenter.estimateFacePose();
    waitKey(0);
}

void testFaceDetection()
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

    BackgroundSegmentation segmenter(face);
    std::vector<cv::Rect> faces;
    if (segmenter.detectFaces(faces)) {
        for (const auto& rect : faces) {
            cv::rectangle(face.image, rect, Scalar(255, 255, 255), 5);
        }
        imshow("image", face.image);
        waitKey(0);
    } else {
        std::cout << "No face detected!" << std::endl;
    }

    face.crop(faces[0]);
    imshow("image", face.image);
    waitKey(0);

    viewPointCloud(face.cloud);
}

void testKmeans()
{

    cv::Mat depth(4, 1, CV_32F);

    depth.at<float>(0, 0) = 0;
    depth.at<float>(1, 0) = 1;
    depth.at<float>(2, 0) = 110;
    depth.at<float>(3, 0) = 109;

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

    string x;
    cin >> x;
}
}

#endif // TEST_H
