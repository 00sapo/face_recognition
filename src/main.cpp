#include <iostream>

#include "test.h"

void testFunctions();

int main()
{

    testFunctions();
    /*
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "004_.*"); // example: loads only .png files starting with 014

    Face face;

    //    loader.setDownscalingRatio(0.5);

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return 0;
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

    viewPointCloud(face.cloud);

    std::cout << "Removing background..." << std::endl;
    segmenter.removeBackground(face);
    std::cout << "Done!" << std::endl;

    viewPointCloud(face.cloud);

    //segmenter.setFace(face);

    std::cout << "Estimating face pose..." << std::endl;
    segmenter.estimateFacePose(face);
    std::cout << "Done!" << std::endl;
    */

    return 0;
}

void testFunctions()
{
    //test::testSingletonSettings();
    //
    //test::testFaceLoader();
    //
    //test::testFindThreshold();
    //
    //test::testGetDepthMap();
    //
    //test::testKmeans();
    //
    //test::testFaceDetection();
    //
    //    test::testDetectFacePose();
    //
    //    test::testEulerAnglesToRotationMatrix();
    //
    //    test::testPoseClustering();
    //
    test::testKmeans2();
}
