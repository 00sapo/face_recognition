#include "covariancecomputer.h"

#include "face.h"
#include "lbp.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using cv::Mat;

namespace face {

CovarianceComputer::CovarianceComputer() {}

vector<std::pair<Mat, Mat>> CovarianceComputer::computeCovarianceRepresentation(const vector<Face>& faces, int subsets) const
{
    cout << "Clustering poses..." << endl;
    auto centers = clusterizePoses(faces, subsets);
    cout << "Assigning faces to clusters..." << endl;
    auto clusters = assignFacesToClusters(faces, centers);

    vector<std::pair<Mat, Mat>> covariances;
    for (const auto& cluster : clusters) {
        Mat imgCovariance, depthCovariance;

        // compute covariance representation of the set
        cout << "Set to covariance..." << endl;
        setToCovariance(cluster, imgCovariance, depthCovariance);
        covariances.emplace_back(imgCovariance, depthCovariance);
    }

    return covariances;
}

Pose CovarianceComputer::eulerAnglesToRotationMatrix(const cv::Vec3f& theta)
{
    // Calculate rotation about x axis
    float cosx = cos(theta[0]);
    float senx = sin(theta[0]);
    float cosy = cos(theta[1]);
    float seny = sin(theta[1]);
    float cosz = cos(theta[2]);
    float senz = sin(theta[2]);

    return Pose(cosy * cosz, cosx * senz + senx * seny * cosz, senx * senz - cosx * seny * cosz,
        -cosy * senz, cosx * cosz - senx * seny * senz, senx * cosz + cosx * seny * senz,
        seny, -senx * cosy, cosx * cosy);
}

vector<Pose> CovarianceComputer::clusterizePoses(const vector<Face>& faces, int numCenters) const
{
    if (faces.size() < numCenters)
        return vector<Pose>(0);

    // retrieve faces' poses
    vector<Pose> poses;
    poses.reserve(faces.size());
    for (const auto& face : faces) {
        poses.push_back(face.getRotationMatrix());
    }

    // clusterize poses
    vector<int> bestLabels;
    cv::Mat centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(poses, numCenters, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.rows != numCenters) {
        std::cout << "Clustering poses failed!" << std::endl;
        return vector<Pose>(numCenters); // TODO: check default Pose value
    }

    // convert from Mat to vector<Pose>
    vector<Pose> ctrs;
    for (int i = 0; i < centers.rows; ++i) {
        ctrs.emplace_back(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2),
            centers.at<float>(i, 3), centers.at<float>(i, 4), centers.at<float>(i, 5),
            centers.at<float>(i, 6), centers.at<float>(i, 7), centers.at<float>(i, 8));
    }
    return ctrs;
}

vector<vector<const Face*>> CovarianceComputer::assignFacesToClusters(const vector<Face>& faces, const vector<Pose>& centers) const
{
    vector<vector<const Face*>> clusters(centers.size());
    for (const auto& face : faces) {
        int index = getNearestCenterId(face.getRotationMatrix(), centers);
        clusters[index].push_back(&face);
    }
    return clusters;
}

int CovarianceComputer::getNearestCenterId(const Pose& pose, const vector<Pose>& centers) const
{
    float min = std::numeric_limits<float>::max();
    int index = 0;
    for (int i = 0; i < centers.size(); ++i) {
        float norm = cv::norm(centers[i], pose, cv::NORM_L2);
        if (norm < min) {
            min = norm;
            index = i;
        }
    }
    return index;
}

void CovarianceComputer::setToCovariance(const vector<const Face*>& set, Mat &imageCovariance, Mat &depthCovariance) const
{
    const int SET_SIZE = set.size();
    if (SET_SIZE == 0) {
        imageCovariance = Mat::zeros(16, 16, CV_32FC1);
        depthCovariance = Mat::zeros(16, 16, CV_32FC1);
        return;
    }

    vector<vector<Mat>> imageBlocks(16);
    vector<vector<Mat>> depthBlocks(16);
    Mat imageMean(16, SET_SIZE, CV_32FC1);
    Mat depthMean(16, SET_SIZE, CV_32FC1);

    for (int i = 0; i < 16; ++i) {
        imageBlocks[i].resize(SET_SIZE);
        depthBlocks[i].resize(SET_SIZE);
    }

    // for each face in the set...
    for (int i = 0; i < SET_SIZE; ++i) {

        if (set[i]->image.empty() || set[i]->depthMap.empty())
            std::cout << "ERROR! Empty image!!" << std::endl;

        // compute 4x4 blocks size
        const auto HEIGHT = set[i]->getHeight();
        const auto WIDTH = set[i]->getWidth();

        const int BLOCK_H = HEIGHT / 4;
        const int BLOCK_W = WIDTH / 4;

        // for each of the 16 blocks of the face...
        for (int y = 0, q = 0; y <= HEIGHT - BLOCK_H; y += BLOCK_H, ++q) {
            for (int x = 0, p = 0; x <= WIDTH - BLOCK_W; x += BLOCK_W, ++p) {

                // crop block region
                cv::Rect roi(x, y, BLOCK_W, BLOCK_H);
                Mat image = set[i]->image(roi);
                Mat depth = set[i]->depthMap(roi);

                // compute LBP of the block
                auto imageHist = OLBPHist(image);
                auto depthHist = OLBPHist(depth);

                imageBlocks[p + 4*q][i] = imageHist;
                depthBlocks[p + 4*q][i] = depthHist;

                imageMean.at<float>(p + 4*q, i) = mean(imageHist)[0];
                depthMean.at<float>(p + 4*q, i) = mean(depthHist)[0];
            }
        }
    }

    std::cout << "Image block size: " << imageBlocks.size() << std::endl;
    std::cout << "Depth block size: " << depthBlocks.size() << std::endl;

    //cv::calcCovarMatrix(imageBlocks, imageCovariance, imageMean, cv::COVAR_NORMAL, CV_32FC1);
    //cv::calcCovarMatrix(depthBlocks, depthCovariance, depthMean, cv::COVAR_NORMAL, CV_32FC1);

    imageCovariance = Mat(16,16,CV_32FC1);
    depthCovariance = Mat(16,16,CV_32FC1);

    for (int p = 0; p < 16; ++p) {
        for (int q = 0; q < 16; ++q) {
            float imageValue = 0, depthValue = 0;
            for (int i = 0; i < SET_SIZE; ++i) {
                imageValue += (imageBlocks[p][i] - imageMean.at<float>(p,i)).dot(imageBlocks[q][i] - imageMean.at<float>(q,i));
                depthValue += (depthBlocks[p][i] - depthMean.at<float>(p,i)).dot(depthBlocks[q][i] - depthMean.at<float>(q,i));
            }
            imageCovariance.at<float>(p,q) = imageValue / SET_SIZE;
            depthCovariance.at<float>(p,q) = depthValue / SET_SIZE;
        }
    }

    return;
}

/*
void PoseManager::addPoseData(const Pose &pose)
{
    posesData.push_back(pose);
}
*/

} // namespace face
