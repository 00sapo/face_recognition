#ifndef FACE_FACERECOGNIZER_H
#define FACE_FACERECOGNIZER_H

#include <svmmanager.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "svmstein.h"

namespace face {

class SVMTrainer : SVMManager {
public:
    SVMTrainer(int c = 3);

    //    SVMTrainer(const std::string& fileName);

    /**
     * @brief Given a vector of faces trains an SVM model to recognize those faces
     * @param trainingSamples: a set of vectors each one containing faces of a different person
     * @param labels: labels to be assigned to each identity; these labels will be returned
     *                by FaceRecognizer::predict() when identifies a person
     */
    void train(const Image4DSetComponentMatrix& trainingSamples,
        const std::vector<std::string>& samplLabels = std::vector<std::string>());

    /**
     * @brief saves a trained model
     * @param fileName: path to save the model to
     * @return true if saved as expected
     */
    bool save(const std::string& directoryName);

private:
    void trainSVMs(cv::Mat& data, const std::vector<int>& indexes, ImgType svmToTrain);
    cv::Mat removeRows(cv::Mat& data, cv::Mat& removed, int baseIdIndex, int subset) const;
    void restoreRows(cv::Mat& data, cv::Mat& removed, int baseIdIndex, int subset) const;
    Mat formatDataForTraining(const MatMatrix& data, std::vector<int>& indexes);
};

} // namespace face

#endif // FACE_FACERECOGNIZER_H
