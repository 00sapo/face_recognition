#ifndef FACE_FACERECOGNIZER_H
#define FACE_FACERECOGNIZER_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "svmstein.h"

namespace face {

class Face;

class NotImplementedException : public std::logic_error {
public:
    NotImplementedException()
        : std::logic_error("Function not yet implemented")
    {
    }
};

using FaceMatrix = std::vector<std::vector<face::Face>>;
using MatMatrix = std::vector<std::vector<cv::Mat>>;
using SVMSteinMatrix = std::vector<std::vector<SVMStein>>;

class FaceRecognizer {
public:
    static const std::string unknownIdentity; // unknown identity label

    FaceRecognizer(int c = 3);

    FaceRecognizer(const std::string& fileName);

    /**
     * @brief Given a vector of faces trains an SVM model to recognize those faces
     * @param trainingSamples: a set of vectors each one containing faces of a different person
     * @param labels: labels to be assigned to each identity; these labels will be returned
     *                by FaceRecognizer::predict() when identifies a person
     */
    void train(const MatMatrix& grayscaleCovar, const MatMatrix& depthmapCovar,
        const std::vector<std::string>& samplLabels = std::vector<std::string>());

    /**
     * @brief predict predicts the identity of the given face set
     * @param identity: various Faces of the same person
     * @return person label if the identity was in the training samples and is recognized,
     *         unknownIdentity otherwise
     */
    std::string predict(const std::vector<cv::Mat>& grayscaleCovar, const std::vector<cv::Mat>& depthmapCovar) const;

    /**
     * @brief loads a pretrained model
     * @param fileName: path of the folder containing the pretrained models
     * @return true if succedes
     */
    bool load(const std::string& directoryName);

    /**
     * @brief saves a trained model
     * @param fileName: path to save the model to
     * @return true if saved as expected
     */
    bool save(const std::string& directoryName);

private:
    enum class ImgType {
        grayscale,
        depthmap
    };

    int c = 3; // number of head rotation subsets for each identity
    int N = 0; // number of identities provided for training
    std::vector<std::string> IDs; // labels associated to each identity in the same order as in grayscaleSVMs and depthmapSVMs
    std::vector<std::vector<SVMStein>> grayscaleSVMs; // a row for each identity and a column for each head rotation subset
    std::vector<std::vector<SVMStein>> depthmapSVMs; // thus resulting in a Nxc matrix where N is the number of identities
        // and c the number of head rotation subsets

    void trainSVMs(const cv::Mat& data, ImgType svmToTrain);

    /**
     * @brief formatDataForTraining transforms the input dataset in a suitable format to be used by
     *        grayscaleSVMs and depthmapSVMs
     * @param data: vector of identities. For each identity it contains a vector of covariance matrixes
     * @param indexes: a vector with the starting rows indexes for each identity (with length = dataIn.size())
     *
     * @return formatted data to be feed into SVMModel. It has one row for each Mat contained in dataIn
     *         and a number of columns equal to Mat::rows x Mat::columns (assuming every Mat in dataIn
     *         has the same dimensions)
     */
    cv::Mat formatDataForTraining(const MatMatrix& data) const;

    cv::Mat formatDataForPrediction(const std::vector<cv::Mat>& data) const;
};

} // namespace face

#endif // FACE_FACERECOGNIZER_H
