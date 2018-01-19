#include "covariancecomputer.h"

#include "face.h"
#include "utils.h"

using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace face {

// FIXME: not a good name, it's not a matrix because it can have rows of different sizes
using MatMatrix = vector<vector<Mat>>;

namespace covariance {

    /**
 * @brief getNearestCenterId
 * @param poseEstimation
 * @return id of the nearest center to the input pose estimation
 */
    int getNearestCenterId(const cv::Vec3f& pose, const vector<float>& centers);

    void getNormalizedCovariances(const vector<Face>& identity, int subsets, vector<Mat>& grayscaleCovarOut,
        vector<Mat>& depthmapCovarOut)
    {
        grayscaleCovarOut.clear();
        depthmapCovarOut.clear();

        auto pairs = computeCovarianceRepresentation(identity, subsets);
        for (const auto& pair : pairs) {
            Mat normalizedGrayscale, normalizedDepthmap;
            cv::normalize(pair.first, normalizedGrayscale);
            cv::normalize(pair.second, normalizedDepthmap);
            grayscaleCovarOut.push_back(normalizedGrayscale);
            depthmapCovarOut.push_back(normalizedDepthmap);
        }
    }

    vector<std::pair<Mat, Mat>> computeCovarianceRepresentation(const vector<Face>& faces, int subsets)
    {
        auto centers = clusterizePoses(faces, subsets);
        auto clusters = assignFacesToClusters(faces, centers);

        vector<std::pair<Mat, Mat>> covariances;
        for (const auto& cluster : clusters) {
            Mat imgCovariance, depthCovariance;

            // compute covariance representation of the set
            setToCovariance(cluster, imgCovariance, depthCovariance);
            covariances.emplace_back(imgCovariance, depthCovariance);
        }

        return covariances;
    }

    vector<float> clusterizePoses(const vector<Face>& faces, int numCenters)
    {
        if (faces.size() < numCenters)
            return vector<float>(0);

        // retrieve faces' poses
        vector<float> poses;
        poses.reserve(faces.size());
        for (const auto& face : faces) {
            poses.push_back(face.getEulerAngles()[1]);
        }

        // clusterize poses
        vector<int> bestLabels;
        cv::Mat centers;
        cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
        cv::kmeans(poses, numCenters, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

        if (centers.rows != numCenters) {
            std::cout << "Clustering poses failed!" << std::endl;
            return vector<float>(numCenters); // TODO: check default Pose value
        }

        // convert from Mat to vector<float>
        vector<float> ctrs;
        for (int i = 0; i < centers.rows; ++i) {
            ctrs.push_back(centers.at<float>(i));
        }

        std::sort(begin(ctrs), end(ctrs));

        return ctrs;
    }

    vector<vector<const Face*>> assignFacesToClusters(const vector<Face>& faces, const vector<float>& centers)
    {
        vector<vector<const Face*>> clusters(centers.size());
        for (const auto& face : faces) {
            int index = getNearestCenterId(face.getEulerAngles(), centers);
            clusters[index].push_back(&face);
        }

        return clusters;
    }

    int getNearestCenterId(const cv::Vec3f& pose, const vector<float>& centers)
    {
        float min = std::numeric_limits<float>::max();
        int index = 0;
        for (size_t i = 0; i < centers.size(); ++i) {
            float distance = std::abs(pose[1] - centers[i]);
            if (distance < min) {
                min = distance;
                index = i;
            }
        }
        return index;
    }

    void setToCovariance(const vector<const Face*>& set, Mat& imageCovariance, Mat& depthCovariance)
    {
        int SET_SIZE = set.size();
        if (SET_SIZE == 0) {
            imageCovariance = Mat::zeros(16, 16, CV_32FC1);
            depthCovariance = Mat::zeros(16, 16, CV_32FC1);
            return;
        }

        MatMatrix imageBlocks(16);
        MatMatrix depthBlocks(16);
        //for (int i = 0; i < 16; ++i) {
        //    imageBlocks[i].resize(SET_SIZE);
        //    depthBlocks[i].resize(SET_SIZE);
        //}

        Mat imageMean(16, SET_SIZE, CV_32FC1);
        Mat depthMean(16, SET_SIZE, CV_32FC1);

        // for each face in the set...
        int i = 0;
        for (const auto& face : set) {

            assert(!face->image.empty() && !face->depthMap.empty()
                && "ERROR! Empty image!!");

            // compute 4x4 blocks size
            const auto HEIGHT = face->getHeight();
            const auto WIDTH = face->getWidth();
            const auto BLOCK_H = HEIGHT / 4;
            const auto BLOCK_W = WIDTH / 4;

            if (BLOCK_H <= 2 || BLOCK_W <= 2)
                continue;

            // for each of the 16 blocks of the face...
            for (size_t y = 0, q = 0; q < 4; y += BLOCK_H, ++q) {
                for (size_t x = 0, p = 0; p < 4; x += BLOCK_W, ++p) {

                    // crop block region
                    cv::Rect roi(x, y, BLOCK_W, BLOCK_H);
                    Mat image = face->image(roi);
                    Mat depth = face->depthMap(roi);

                    // compute LBP of the block
                    auto imageHist = OLBPHist(image);
                    auto depthHist = OLBPHist(depth);

                    imageBlocks[p + 4 * q].push_back(imageHist);
                    depthBlocks[p + 4 * q].push_back(depthHist);

                    imageMean.at<float>(p + 4 * q, i) = mean(imageHist)[0];
                    depthMean.at<float>(p + 4 * q, i) = mean(depthHist)[0];
                }
            }
            ++i;
        }
        SET_SIZE = i;
        depthMean.resize(SET_SIZE);
        imageMean.resize(SET_SIZE);

        imageCovariance = Mat(16, 16, CV_32FC1);
        depthCovariance = Mat(16, 16, CV_32FC1);

        // Computing covariances.
        // OpenCV cv::calcCovarMatrix() could be used but it's very hard to obtain the same result
        // because data representation shold be changed requiring extra coding and runtime work
        for (int p = 0; p < 16; ++p) {
            for (int q = 0; q < 16; ++q) {
                float imageValue = 0, depthValue = 0;
                for (int i = 0; i < SET_SIZE; ++i) {
                    imageValue += (imageBlocks[p][i] - imageMean.at<float>(p, i)).dot(imageBlocks[q][i] - imageMean.at<float>(q, i));
                    depthValue += (depthBlocks[p][i] - depthMean.at<float>(p, i)).dot(depthBlocks[q][i] - depthMean.at<float>(q, i));
                }
                imageCovariance.at<float>(p, q) = imageValue / SET_SIZE;
                depthCovariance.at<float>(p, q) = depthValue / SET_SIZE;
            }
        }

        return;
    }

} // namespace covariance

} // namespace face
