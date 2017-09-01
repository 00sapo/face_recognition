#include "covariancecomputer.h"

#include "face.h"
#include "lbp.h"


using std::string;
using std::vector;
using cv::Mat;

namespace face {

CovarianceComputer::CovarianceComputer() { }


vector<std::pair<Mat,Mat>> CovarianceComputer::computeCovarianceRepresentation(const vector<Face> &faces, int subsets) const
{
    auto centers  = clusterizePoses(faces, subsets);
    auto clusters = assignFacesToClusters(faces, centers);

    vector<std::pair<Mat,Mat>> covariances;
    for (const auto &cluster : clusters) {
        Mat imgCovariance, depthCovariance;
        setToCovariance(cluster, imgCovariance, depthCovariance);
        covariances.emplace_back(imgCovariance, depthCovariance);
    }

    return covariances;
}


Pose CovarianceComputer::eulerAnglesToRotationMatrix(const cv::Vec3f &theta)
{
    // Calculate rotation about x axis
    float cosx = cos(theta[0]);
    float senx = sin(theta[0]);
    float cosy = cos(theta[1]);
    float seny = sin(theta[1]);
    float cosz = cos(theta[2]);
    float senz = sin(theta[2]);

    return Pose( cosy * cosz, cosx * senz + senx * seny * cosz, senx * senz - cosx * seny * cosz,
                -cosy * senz, cosx * cosz - senx * seny * senz, senx * cosz + cosx * seny * senz,
                 seny, -senx * cosy, cosx * cosy);
}


vector<Pose> CovarianceComputer::clusterizePoses(const vector<Face> &faces, int numCenters) const
{
    if (faces.size() < numCenters)
        return vector<Pose>(0);

    vector<Pose> poses;
    poses.reserve(faces.size());
    for (const auto &face : faces) {
        poses.push_back(face.getRotationMatrix());
    }

    vector<int> bestLabels;
    cv::Mat centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(poses, numCenters, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.rows != numCenters) {
        std::cout << "Clustering poses failed!" << std::endl;
        return vector<Pose>(numCenters);        // TODO: check default Pose value
    }

    vector<Pose> ctrs;
    for (int  i = 0; i < centers.rows; ++i) {
        ctrs.emplace_back(centers.at<float>(i,0), centers.at<float>(i,1), centers.at<float>(i,2),
                          centers.at<float>(i,3), centers.at<float>(i,4), centers.at<float>(i,5),
                          centers.at<float>(i,6), centers.at<float>(i,7), centers.at<float>(i,8));
    }
    return centers;
}

vector<vector<const Face*>> CovarianceComputer::assignFacesToClusters(const vector<Face> &faces, const vector<Pose> &centers) const
{
   vector<vector<const Face*>> clusters(centers.size());
   for (const auto &face : faces) {
       int index = getNearestCenterId(face.getRotationMatrix(), centers);
       clusters[index].push_back(&face);
   }
   return clusters;
}

int CovarianceComputer::getNearestCenterId(const Pose &pose, const vector<Pose> &centers) const
{
    cv::Mat poseMat(pose);
    float min = std::numeric_limits<float>::max();
    int index = 0;
    for (int i = 0; i < centers.size(); ++i) {
        float norm = cv::norm(centers[i].t(), poseMat, cv::NORM_L2);
        if (norm < min) {
            min = norm;
            index = i;
        }
    }
    return index;
}

void CovarianceComputer::setToCovariance(const vector<const Face*> &set, Mat &imageCovariance, Mat &depthCovariance) const
{
    const int SET_SIZE = set.size();
    if (SET_SIZE == 0) {
        imageCovariance = Mat::zeros(16,16, CV_32FC1);
        depthCovariance = Mat::zeros(16,16, CV_32FC1);
        return;
    }

    const auto HEIGHT = set[0]->getHeight();
    const auto WIDTH  = set[0]->getWidth();

    const int BLOCK_H = HEIGHT/4;
    const int BLOCK_W = WIDTH /4;

    vector<vector<Mat>> imageBlocks(SET_SIZE);
    vector<vector<Mat>> depthBlocks(SET_SIZE);
    vector<vector<float>> imageMeans(SET_SIZE);
    vector<vector<float>> depthMeans(SET_SIZE);
    for (int  i = 0; i < SET_SIZE; ++i) {
        for (int x = 0; x < HEIGHT - BLOCK_H; x += BLOCK_H) {
            for (int y = 0; y < WIDTH - BLOCK_W; y += BLOCK_W) {
                cv::Rect roi(x, y, x + BLOCK_W, y + BLOCK_W);
                auto image = set[i]->image(roi);
                auto depth = set[i]->depthMap(roi);
                auto imageLBP = OLBPHist(image);
                auto depthLBP = OLBPHist(depth);
                imageBlocks[i].push_back(imageLBP);
                depthBlocks[i].push_back(depthLBP);
                imageMeans[i].push_back(HistMean(imageLBP));
                depthMeans[i].push_back(HistMean(depthLBP));
            }
        }
    }

    imageCovariance = Mat(16, 16, CV_32FC1);
    depthCovariance = Mat(16, 16, CV_32FC1);
    for (int p = 0; p < 16; ++p) {
        for (int q = 0; q < 16; ++q) {
            float img_sum = 0, dpt_sum = 0;
            for (int i = 0; i < SET_SIZE; ++i) {
                auto &img_x_p = imageBlocks[i][p];
                auto &img_x_q = imageBlocks[i][q];

                auto x_pNorm = img_x_p - imageMeans[i][p];
                auto x_qNorm = img_x_q - imageMeans[i][q];

                img_sum += x_pNorm.dot(x_qNorm);

                auto &dpt_x_p = depthBlocks[i][p];
                auto &dpt_x_q = depthBlocks[i][q];

                x_pNorm = dpt_x_p - depthMeans[i][p];
                x_qNorm = dpt_x_q - depthMeans[i][q];

                dpt_sum += x_pNorm.dot(x_qNorm);
            }

            imageCovariance.at<float>(p,q) = img_sum / SET_SIZE;
            depthCovariance.at<float>(p,q) = dpt_sum / SET_SIZE;
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

}   // face
