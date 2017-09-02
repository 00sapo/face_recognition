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
    //face::test::testDetectFacePose();
    //
    //test::testLoadSpeed();
    //
    //test::testEulerAnglesToRotationMatrix();
    //
    //test::testPoseClustering();
    //
    //test::testKmeans2();
    //
    face::test::covarianceTest();

    cout << "\n\nTests finished!" << endl;
}
