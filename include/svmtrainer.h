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
     */
    void train(Image4DComponent* trainingSamples);

    /**
     * @brief saves a trained model
     * @param fileName: path to save the model to
     * @return true if saved as expected
     */
    bool save(const std::string& directoryName);

private:
    void trainSVMs(vector<Mat> &data, ImgType svmToTrain);
    cv::Mat removeRow(cv::Mat& data, cv::Mat& removed, int id) const;
    void restoreRow(cv::Mat& data, cv::Mat& removed, int id) const;
    vector<Mat> formatDataForTraining(const MatSet& data);
};

} // namespace face

#endif // FACE_FACERECOGNIZER_H
