#ifndef COVARIANCE_TEST_H
#define COVARIANCE_TEST_H

#include "face.h"
#include "faceloader.h"
#include "lbp.h"

using namespace cv;
using namespace std;

void splitImage()
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

    int croppedWidth = face.getWidth() / 4;
    int croppedHeight = face.getHeight() / 4;
    for (uint y = 0; y < face.getHeight(); y += croppedHeight) {
        for (uint x = 0; x < face.getWidth(); x += croppedWidth) {
            Mat cropped = face.image(Rect(x, y, croppedWidth, croppedHeight));

            Mat LBPHist = OLBPHist(cropped);
            cout << LBPHist << endl;
            imshow("image", cropped);
            waitKey(0);
        }
    }
}

#endif // COVARIANCE_TEST_H
