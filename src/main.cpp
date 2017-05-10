#include <iostream>

#include "test.h"

void testFunctions();

int main()
{

    testFunctions();

    return 0;
}


void testFunctions()  {
    cout << "SingletonSettings test..." << endl;
    //test::testSingletonSettings();

    cout << "\n\nFace loader test..." << endl;
    //test::testFaceLoader();

    cout << "\n\nFind threshold test..." << endl;
    //test::testFindThreshold();

    cout << "\n\nGet depth map test..." << endl;
    //test::testGetDepthMap();

    cout << "\n\nDetect face pose..." << endl;
    //test::testDetectFacePose();

    cout << "\n\nDetect faces test..." << endl;
    test::testFaceDetection();

    cout << "\n\nTests finished!" << endl;
}
