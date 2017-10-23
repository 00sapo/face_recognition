#include <covariancecomputer.h>
#include <facecropper.h>
#include <image4dcomponent.h>
#include <iostream>
#include <kmeansbackgroundremover.h>
#include <poseclusterizer.h>
#include <preprocessorpipe.h>
#include <string.h>
#include <svmtrainer.h>
#include <vector>

#include "image4dloader.h"

using namespace face;
using namespace std;

void training(string trainingSetDir, string outputDir)
{

    string dirPath = trainingSetDir;
    Image4DLoader loader(dirPath, "000_.*");

    PreprocessorPipe pipe;
    Image4DVectorComposite set;
    for (int i = 0; i < 26; ++i) {
        stringstream fileNameRegEx;
        fileNameRegEx << setw(3) << setfill('0') << i << "_.*";

        loader.setFileNameRegEx(fileNameRegEx.str());
        set.add(*loader.get());
    }

    pipe.setImageSet(&set);

    KmeansBackgroundRemover backgroundRemover;
    FaceCropper faceCropper;
    PoseClusterizer poseClusterizer;
    CovarianceComputer covarianceComputer;

    pipe.push_back(backgroundRemover);
    pipe.push_back(faceCropper);
    pipe.push_back(poseClusterizer);
    pipe.push_back(covarianceComputer);
    pipe.processPipe();

    SVMTrainer faceRec;
    faceRec.train(pipe.getImageSet());

    faceRec.save(outputDir);
}

void testing(string testingSetDir, string outputDir)
{
    //TODO
}

int main(int count, char* args[])
{
    if (count != 4) {
        cout << "This software has two commands: " << endl;
        cout << "\t* training <training set directory> <model output directory>\t\t- to train the model" << endl;
        cout << "\t* testing <testing set directory> <model directory>\t\t- to test the model" << endl;
        return 1;
    }
    if (!strncmp(args[1], "training", 8))
        training(args[2], args[3]);
    if (!strncmp(args[1], "testing", 7))
        testing(args[2], args[3]);

    return 0;
}
