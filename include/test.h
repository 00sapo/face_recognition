#ifndef TEST_H
#define TEST_H

#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/visualization/area_picking_event.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "face.h"
#include "faceloader.h"
#include "facesegmenter.h"
#include "posemanager.h"
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

Mat testEulerAnglesToRotationMatrix()
{
    srand(time(NULL));
    float r1 = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2 * M_PI)));
    float r2 = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2 * M_PI)));
    float r3 = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2 * M_PI)));
    Vec3f euler = { r1, r2, r3 };
    PoseManager pm;

    Mat rotation = pm.eulerAnglesToRotationMatrix(euler);

    std::cout << "Euler Angles:" << std::endl;
    std::cout << euler << std::endl;
    std::cout << "Rotation Matrix:" << std::endl;
    std::cout << rotation << std::endl;

    return rotation;
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

    FaceSegmenter segmenter;
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

    FaceSegmenter segmenter;
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

    PoseManager poseManager = PoseManager();

    std::cout << "Estimating face pose..." << std::endl;
    poseManager.estimateFacePose(face);
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

    FaceSegmenter segmenter;
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

void testKmeans2()
{

    Matx<float, 9, 1> R0(0.123, 0.345, 0.987, 0.1842, 0.567, 0.832, 0.324, 0.431, 0.111);
    Matx<float, 9, 1> R1(0.7153, 0.3345, 0.3987, 0.91842, 0.5677, 0.7832, 0.5324, 0.4831, 0.6111);
    Matx<float, 9, 1> R2(0.5123, 0.5345, 0.1987, 0.19842, 0.2567, 0.8832, 0.2324, 0.6431, 0.2111);
    Matx<float, 9, 1> R3(0.3123, 0.4345, 0.5987, 0.91842, 0.3567, 0.8732, 0.9324, 0.8431, 0.9111);
    Matx<float, 9, 1> R4(0.6123, 0.5345, 0.4987, 0.5842, 0.1567, 0.1832, 0.1324, 0.9431, 0.7111);
    Matx<float, 9, 1> R5(0.9123, 0.2345, 0.3987, 0.11842, 0.4567, 0.1832, 0.7324, 0.5431, 0.111);
    Matx<float, 9, 1> R6(0.0123, 0.9345, 0.7987, 0.01842, 0.3567, 0.9832, 0.8324, 0.3431, 0.9111);
    Matx<float, 9, 1> R7(0.6123, 0.0345, 0.0987, 0.31842, 0.0567, 0.0832, 0.7324, 0.5431, 0.6111);
    Matx<float, 9, 1> R8(0.4123, 0.1345, 0.1987, 0.61842, 0.5567, 0.6832, 0.6324, 0.6431, 0.2111);
    Matx<float, 9, 1> R9(0.2123, 0.2345, 0.2987, 0.81842, 0.8567, 0.5832, 0.3324, 0.8431, 0.3111);

    vector<Matx<float, 9, 1>> data = { R0, R1, R2, R3, R4, R5, R6, R7, R8, R9 };
    Matx<float, 2, 9> centers;

    vector<int> bestLabels;
    cout << "Clustering..." << endl;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(data, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
    cout << "Done!" << endl;

    //    cout << "Size: " << centers.size() << endl;
    //cout << "Cols: " << centers.cols << endl;

    cout << "Labels for points..." << endl;
    for (uint i = 0; i < bestLabels.size(); ++i) {
        cout << bestLabels.at(i) << endl;
    }
    cout << "Done!" << endl;

    cout << endl;
    cout << "Centers are: " << endl;
    cout << centers << endl;
    system("read -p 'Press [enter] to continue'");

    cout << "\n\nTests finished!" << endl;
}

void testPoseClustering()
{
    PoseManager pm = PoseManager();
    for (int i = 0; i < 40; i++) {
        Mat pose = testEulerAnglesToRotationMatrix();
        pm.addPoseData(pose);
    }

    pm.clusterizePoses(4);

    Mat pose = testEulerAnglesToRotationMatrix();
    std::cout << "Nearest Center: " << endl;
    std::cout << pm.getNearestCenterId(pose) << endl;
}
}
#endif // TEST_H
