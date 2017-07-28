#include <iostream>

#include "test.h"

void testFunctions();

int main()
{
    testFunctions();

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
    test::testDetectFacePose();
    //
    //test::testEulerAnglesToRotationMatrix();
    //
    //test::testPoseClustering();
    //
    //test::testKmeans2();
    //
    //test::covarianceTest();

    cout << "\n\nTests finished!" << endl;
}
