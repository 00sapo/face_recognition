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
    //test::testImage4DLoader();
    //
    //test::testFindThreshold();
    //
    //test::testGetDepthMap();
    //
    //test::testKmeans();
    //
    //face::test::testPreprocessing();
    //
    //test::testLoadSpeed();
    //
    //test::testEulerAnglesToRotationMatrix();
    //
    //test::testPoseClustering();
    //
    //test::testKmeans2();
    //
    //face::test::covarianceTest();
    face::test::testSVM();
    //        face::test::covarianceTest();
    //
    //    face::test::testBackgroundRemoval();

    cout << "\n\nTests finished!" << endl;
}
