#include <iostream>
#include <preprocessor.h>
#include <string.h>
#include <svmtrainer.h>
#include <vector>

#include "face.h"
#include "image4dloader.h"

using namespace face;
using namespace std;

using Image4DMatrix = std::vector<std::vector<Image4D>>;

void training(string trainingSetDir, string outputDir)
{

    string dirPath = trainingSetDir;
    Image4DLoader loader(dirPath, "000_.*");

    Image4DMatrix identities;
    for (int i = 0; i <= 25; ++i) {
        string fileNameRegEx = i / 10 >= 1 ? "0" : "00";
        fileNameRegEx += std::to_string(i) + "_.*";

        loader.setFileNameRegEx(fileNameRegEx);
        identities.push_back(loader.get());
    }

    Preprocessor preproc;

    int i = 0;
    FaceMatrix peoples;
    for (auto& id : identities) {
        cout << "Preprocessing images of person " << i++ << endl;
        peoples.push_back(preproc.preprocess(id));
    }

    SVMTrainer faceRec;
    faceRec.train(peoples);

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
