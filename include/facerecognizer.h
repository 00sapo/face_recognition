#ifndef FACE_FACERECOGNIZER_H
#define FACE_FACERECOGNIZER_H

#include <vector>

#include "face.h"
#include "covariancecomputer.h"
#include "svmmodel.h"

namespace face {

class FaceRecognizer
{
public:

    static const std::string unknownIdentity;   // unknown identity label

    FaceRecognizer();


    /**
     * @brief Given a vector of faces trains an SVM model to recognize those faces
     * @param trainingSamples: a set of vectors each one containing faces of a different person
     * @param labels: labels to be assigned to each identity; these labels will be returned
     *                by FaceRecognizer::predict() when identifies a person
     */
    void train(const std::vector<std::vector<Face>> &trainingSamples,
               const std::vector<std::string> &samplLabels = std::vector<std::string>());

    /**
     * @brief predict predicts the identity of the given face set
     * @param identity: various Faces of the same person
     * @return person label if the identity was in the training samples and is recognized,
     *         unknownIdentity otherwise
     */
    std::string predict(const std::vector<Face> &identity) const;

    /**
     * @brief loads a pretrained model
     * @param fileName: path of the folder containing the pretrained models
     * @return true if succedes
     */
    bool load(const std::string &fileName);

    /**
     * @brief saves a trained model
     * @param fileName: path to save the model to
     * @return true if saved as expected
     */
    bool save(const std::string &fileName);

private:
    const int c = 3;  // number of head rotation subsets for each identity
    CovarianceComputer covarComputer;
    std::vector<std::string> IDs;  // labels associated to each identity in the same order as in grayscaleSVMs and depthmapSVMs
    std::vector<std::vector<SVMModel>> grayscaleSVMs;   // a row for each identity and a column for each head rotation subset
    std::vector<std::vector<SVMModel>> depthmapSVMs;    // thus resulting in a Nxc matrix where N is the number of identities
                                                        // and c the number of head rotation subsets

    std::vector<std::string> generateLabels(int numOfLabels);

    /**
     * @brief formatDataForTraining transforms the input dataset in a suitable format to be used by
     *        grayscaleSVMs and depthmapSVMs
     * @param dataIn: vector of identities. For each identity it contains a vector of covariance matrixes
     * @param dataOut: formatted data to be feed into SVMModel. It has one row for each Mat contained in dataIn
     *        and a number of columns equal to Mat::rows x Mat::columns (assuming every Mat in dataIn
     *        has the same dimensions)
     * @return a vector with the ranges of rows belonging to each identity (with length = dataIn.size())
     */
    std::vector<cv::Range> formatDataForTraining(const std::vector<std::vector<cv::Mat>> &dataIn, cv::Mat &dataOut) const;
};

}   // namespace face

#endif // FACE_FACERECOGNIZER_H
