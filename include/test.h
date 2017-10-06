#ifndef FACE_TEST_TEST_H
#define FACE_TEST_TEST_H

#include <chrono>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include "covariancecomputer.h"
#include "face.h"
#include "image4d.h"
#include "image4dloader.h"
#include "lbp.h"
#include "preprocessor.h"
#include "settings.h"
#include "svmmodel.h"
#include <numeric>

using cv::Mat;
using cv::waitKey;
using std::cout;
using std::endl;
using std::list;
using std::string;
using std::vector;

namespace face {

namespace test {

    void testSVM()
    {
        string dirPath = "../RGBD_Face_dataset_training/";
        Image4DLoader loader(dirPath, "000_.*");

        vector<vector<Image4D>> identities;
        for (int i = 0; i <= 25; ++i) {
            string fileNameRegEx = i / 10 >= 1 ? "0" : "00";
            fileNameRegEx += std::to_string(i) + "_.*";

            loader.setFileNameRegEx(fileNameRegEx);
            identities.push_back(loader.get());
        }

        Preprocessor preproc;
        cout << "Preprocessing..." << endl;

        vector<vector<Face>> persons;
        for (auto& id : identities) {
            persons.push_back(preproc.preprocess(id));
        }
        vector<Mat> trainingSet;
        int numPoseClusters = 3;
        CovarianceComputer covar;

        vector<vector<std::list<const Face*>>> dataSet;
        for (auto& faces : persons) {

            cout << "Clustering poses..." << endl;
            vector<Pose> centers = covar.clusterizePoses(faces, numPoseClusters);
            cout << "Assigning faces to clusters..." << endl;
            dataSet.push_back(covar.assignFacesToClusters(faces, centers));
        }

        cout << "Training SVM model for 1st cluter of 1st person using leave-one-out per each photo" << endl;
        //leave-one-out for each cluster
        Mat normalizedDepthCovar;
        for (auto& person : dataSet) {

            for (auto& cluster : person) {
                Mat targetImageMatrix, targetDepthCovar;

                for (list<const Face*>::iterator it = cluster.begin(); it != cluster.end(); it++) {
                    const Face* face = *it;
                    cluster.erase(it);

                    vector<std::pair<Mat, Mat>> pairs = covar.computeCovarianceRepresentation(person);
                    for (auto& pair : pairs) {
                        trainingSet.push_back(pair.first);
                    }

                    covar.setToCovariance(std::list<const Face*>{ face }, targetImageMatrix, targetDepthCovar);
                    cv::normalize(targetDepthCovar, normalizedDepthCovar);

                    cout << "Creating SVM model..." << endl;
                    SVMmodel model;
                    cout << "Training model..." << endl;
                    model.train(trainingSet, 0, 0); //1st person, 1st cluster
                    //TODO: predict the normalizedDepthCovar and if it is right increase or decrease true/false positives/negatives
                    cout << "Choosing best parameters" << endl;
                    if (optimalParams.fmeasure >= bestParams[0].fmeasure)
                        bestParams = vector<SteinKernelParams>{ optimalParams };
                    else if (optimalParams.fmeasure == bestParams[0].fmeasure)
                        bestParams.push_back(optimalParams);

                    cluster.insert(it, face);
                }

                cout << "Model trained, computing mean of best parameters are the following" << endl;
                double gamma = 0, C = 0;

                for (auto& param : bestParams) {
                    C += param.C;
                    gamma += param.gamma;
                }
                SteinKernelParams best(C / bestParams.size(), gamma / bestParams.size());

                std::cout << "score obtained by avaraging best parameters: <to be implemented>" << std::endl;
                cout << "C: " << best.C << endl;
                cout << "gamma: " << best.gamma << endl;
                poseModelParams.push_back(best);
            }

            models.push_back(poseModelParams);
        }
    }

    cv::Vec3f randomEulerAngle()
    {
        float r1 = float(rand()) / (float(RAND_MAX) / (2.0f * M_PI));
        float r2 = float(rand()) / (float(RAND_MAX) / (2.0f * M_PI));
        float r3 = float(rand()) / (float(RAND_MAX) / (2.0f * M_PI));
        return { r1, r2, r3 };
    }

    void testSettings()
    {
        cout << "SingletonSettings test..." << endl;
        auto& settings = Settings::getInstance();
        cout << settings.getD() << endl
             << settings.getK() << endl
             << settings.getP() << endl
             << settings.getR() << endl
             << settings.getHeight() << endl
             << settings.getWidth() << endl;
        system("read -p 'Press [enter] to continue'");
    }

    void testImage4DLoader()
    {
        cout << "\n\nFace loader test..." << endl;
        string dirPath = "../RGBD_Face_dataset_training/";
        Image4DLoader loader(dirPath, "014.*");
        auto begin = std::chrono::high_resolution_clock::now();
        auto faceSequence = loader.get();
        if (faceSequence.empty()) {
            cout << "Error loading face!" << endl;
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout << "\n\nFaces loaded!" << endl;
        cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << endl;

        cv::namedWindow("image", cv::WINDOW_NORMAL);
        for (const auto& face : faceSequence) {
            imshow("image", face.image);
            while (waitKey(0) != 'm') {
            }

            //viewPointCloud(face.depthMap);
            imshow("Depth map", face.depthMap);
            waitKey(0);
        }
        system("read -p 'Press [enter] to continue'");
    }

    Pose testEulerAnglesToRotationMatrix()
    {
        srand(time(NULL));
        cv::Vec3f euler = randomEulerAngle();

        Pose rotation = CovarianceComputer::eulerAnglesToRotationMatrix(euler);

        cout << "Euler Angles:" << endl;
        cout << euler << endl;
        cout << "Rotation Matrix:" << endl;
        cout << rotation << endl;

        return rotation;
    }

    void testGetDepthMap()
    {
        cout << "\n\nGet depth map test..." << endl;
        string dirPath = "../RGBD_Face_dataset_training/";
        Image4DLoader loader(dirPath, "014.*"); // example: loads only .png files starting with 014

        Image4D face;

        if (!loader.get(face)) {
            cout << "Failed loading face" << endl;
            return;
        }

        cout << "Face loaded!" << endl;

        Mat depthMap = face.get3DImage();

        imshow("Depth Map", depthMap);
        waitKey(0);
        system("read -p 'Press [enter] to continue'");
    }

    void testPreprocessing()
    {
        cout << "\n\nDetect face pose..." << endl;
        Image4DLoader loader("../RGBD_Face_dataset_training/", ".*");
        auto image4d = loader.get();

        Preprocessor prep;
        auto begin = std::chrono::high_resolution_clock::now();
        auto faces = prep.preprocess(image4d);
        auto end = std::chrono::high_resolution_clock::now();
        cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s" << endl;

        for (auto& face : faces) {
            imshow("Cropped face", face.image);
            cv::waitKey(0);
        }
    }

    void testBackgroundRemoval()
    {
        cout << "\n\nDetect face pose..." << endl;
        Image4DLoader loader("../RGBD_Face_dataset_training/", "000.*");

        auto images = loader.get();
        if (images.empty()) {
            cout << "Failed loading faces" << endl;
            return;
        }

        cout << "Faces loaded!" << endl;

        //        for (auto& image4d : images) {
        //            imshow("Original image", image4d.depthMap);
        //            cv::waitKey(0);
        //        }

        Preprocessor prep;
        prep.segment(images);

        for (auto& image4d : images) {
            imshow("Original image", image4d.depthMap);
            cv::waitKey(0);
        }

        system("read -p 'Press [enter] to continue'");
    }

    void testLoadSpeed()
    {
        cout << "\n\nTest load speed..." << endl;
        Image4DLoader loader("../RGBD_Face_dataset_training/", ".*");

        auto start = std::chrono::high_resolution_clock::now();
        loader.get();
        auto end = std::chrono::high_resolution_clock::now();

        cout << "Faces loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
             << "ms" << endl;

        loader = Image4DLoader("../RGBD_Face_dataset_training/", ".*");
        start = std::chrono::high_resolution_clock::now();
        Image4D image;
        while (loader.hasNext())
            loader.get(image);
        end = std::chrono::high_resolution_clock::now();

        cout << "Faces loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
             << "ms" << endl;
    }

    void testKmeans()
    {
        cout << "\n\nKmeans test..." << endl;
        vector<float> depth;

        depth.push_back(1.543);
        depth.push_back(1.563);
        depth.push_back(1.547);
        depth.push_back(1.743);
        depth.push_back(5.543);
        depth.push_back(5.673);
        depth.push_back(1.915);
        depth.push_back(1.543);
        depth.push_back(5.563);
        depth.push_back(4.743);
        depth.push_back(6.542);
        depth.push_back(1.246);

        Mat centers(1, 2, CV_32F);
        std::vector<int> bestLabels;
        cout << "Clustering..." << endl;
        cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
        cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
        cout << "Done!" << endl;

        cout << "Size: " << centers.size() << endl;
        //std::cout << "Cols: " << centers.cols << std::endl;

        cout << "Labels for points..." << endl;
        for (uint i = 0; i < bestLabels.size(); ++i) {
            cout << bestLabels.at(i) << endl;
        }
        cout << "Done!" << endl;

        system("read -p 'Press [enter] to continue'");
    }

    void testKmeans2()
    {

        Pose R0(0.123, 0.345, 0.987, 0.1842, 0.567, 0.832, 0.324, 0.431, 0.111);
        Pose R1(0.7153, 0.3345, 0.3987, 0.91842, 0.5677, 0.7832, 0.5324, 0.4831, 0.6111);
        Pose R2(0.5123, 0.5345, 0.1987, 0.19842, 0.2567, 0.8832, 0.2324, 0.6431, 0.2111);
        Pose R3(0.3123, 0.4345, 0.5987, 0.91842, 0.3567, 0.8732, 0.9324, 0.8431, 0.9111);
        Pose R4(0.6123, 0.5345, 0.4987, 0.5842, 0.1567, 0.1832, 0.1324, 0.9431, 0.7111);
        Pose R5(0.9123, 0.2345, 0.3987, 0.11842, 0.4567, 0.1832, 0.7324, 0.5431, 0.111);
        Pose R6(0.0123, 0.9345, 0.7987, 0.01842, 0.3567, 0.9832, 0.8324, 0.3431, 0.9111);
        Pose R7(0.6123, 0.0345, 0.0987, 0.31842, 0.0567, 0.0832, 0.7324, 0.5431, 0.6111);
        Pose R8(0.4123, 0.1345, 0.1987, 0.61842, 0.5567, 0.6832, 0.6324, 0.6431, 0.2111);
        Pose R9(0.2123, 0.2345, 0.2987, 0.81842, 0.8567, 0.5832, 0.3324, 0.8431, 0.3111);

        vector<Pose> data = { R0, R1, R2, R3, R4, R5, R6, R7, R8, R9 };
        Mat centers;

        vector<int> bestLabels;
        cout << "Clustering..." << endl;
        cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
        cv::kmeans(data, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
        cout << "Done!" << endl;

        cout << "Size: " << centers.size() << endl;
        cout << "Cols: " << centers.cols << endl;

        cout << "Labels for points..." << endl;
        for (uint i = 0; i < bestLabels.size(); ++i) {
            cout << bestLabels.at(i) << endl;
        }
        cout << "Done!" << endl;

        cout << endl;
        cout << "Centers are: " << endl;
        cout << centers << endl;
        system("read -p 'Press [enter] to continue'");

        cout << "\n\nTests finished!" << endl;
    }

    /*
void testPoseClustering()
{
    srand(time(NULL));
    PoseManager pm;
    for (int i = 0; i < 40; i++) {
        Pose pose = PoseManager::eulerAnglesToRotationMatrix(randomEulerAngle());
        pm.addPoseData(pose);
    }

    pm.clusterizePoses(4);

    Pose pose = PoseManager::eulerAnglesToRotationMatrix(randomEulerAngle());
    cout << "Nearest Center: " << endl;
    cout << pm.getNearestCenterId(pose) << endl;
}
*/

    void covarianceTest()
    {
        cout << "\n\nDetect faces test..." << endl;
        string dirPath = "../RGBD_Face_dataset_training/";
        Image4DLoader loader(dirPath, "000_.*");
        Preprocessor preproc;
        CovarianceComputer covComp;

        cout << "Loading images..." << endl;
        auto images4d = loader.get();
        cout << "Loaded " << images4d.size() << " images" << endl;

        for (const auto& image4d : images4d) {
            cv::imshow("Image", image4d.image);
            cv::imshow("Depth map", image4d.depthMap);
            cv::waitKey(0);
        }

        cout << "Preprocessing 4D images..." << endl;
        auto faces = preproc.preprocess(images4d);
        cout << "Extracted " << faces.size() << " faces from 4D images" << endl;

        for (const auto& face : faces) {
            cv::imshow("Image", face.image);
            cv::imshow("Depth map", face.depthMap);
            cv::waitKey(0);
        }

        /*
        cout << "Computing covariances..." << endl;
        auto covariance = covComp.computeCovarianceRepresentation(faces, 3);

        for (auto& cov : covariance) {
            cv::imshow("Image", cov.first);
            cv::imshow("Depth", cov.second);
            cv::waitKey(0);
        }
        */
    }

} // namespace test
} // namespace face
#endif // TEST_H
