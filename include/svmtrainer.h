#ifndef FACE_FACERECOGNIZER_H
#define FACE_FACERECOGNIZER_H

#include <svmmanager.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "svmstein.h"

namespace face {

class SVMTrainer : SVMManager {
public:
    SVMTrainer();

    //    SVMTrainer(const std::string& fileName);

    /**
     * @brief Given a vector of faces trains an SVM model to recognize those faces
     * @param trainingSamples: a set faces of a different person, clusterized by pose and with
     *                         and normalized covariances matrix
     * @param labels: labels to be assigned to each identity; these labels will be returned
     *                by FaceRecognizer::predict() when identifies a person
     */
    void train(Image4DComponent& trainingSamples, const vector<string>& samplIDs);

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
    Mat formatDataForTraining(const MatSet& data, std::vector<int>& indexes);
};

} // namespace face

#endif // FACE_FACERECOGNIZER_H
