#ifndef SVMTESTER_H
#define SVMTESTER_H
#include "svmstein.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <svmmanager.h>
#include <vector>

using std::string;
namespace face {

class SVMTester : SVMManager {
public:
    static const std::string unknownIdentity; // unknown identity label

    SVMTester(int c = 3);

    /**
     * @brief predict predicts the identity of the given face set
     * @param identity: various Faces of the same person
     * @return person label if the identity was in the training samples and is recognized,
     *         unknownIdentity otherwise
     */
    std::string predict(const Image4DComponent& identity) const;

    /**
     * @brief loads a pretrained model
     * @param fileName: path of the folder containing the pretrained models
     * @return true if succedes
     */
    bool load(const string& directoryName);

private:
    Mat formatDataForPrediction(const vector<Mat>& data) const;
};
}
#endif // SVMTESTER_H
