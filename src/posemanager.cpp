#include "posemanager.h"

#include "face.h"


using std::string;
using std::vector;
using cv::Mat;
using cv::Vec3f;

namespace face {

PoseManager::PoseManager() { }

/*
PoseManager::PoseManager(const std::vector<Face> &faces)
{
    for (const auto &face : faces) {
        addPoseData(face.getRotationMatrix());
    }
}
*/

vector<Mat> PoseManager::computeCovarianceRepresentation(const vector<Face> &faces, int subsets)
{
    auto centers = clusterizePoses(faces, subsets);
    auto clusters = assignFacesToClusters(faces, centers);

    vector<Mat> covariances;
    for (const auto &cluster : clusters) {
        covariances.push_back(setToCovariance(cluster));
    }

    return covariances;
}


Pose PoseManager::eulerAnglesToRotationMatrix(const cv::Vec3f &theta)
{
    // Calculate rotation about x axis
    float cosx = cos(theta[0]);
    float senx = sin(theta[0]);
    float cosy = cos(theta[1]);
    float seny = sin(theta[1]);
    float cosz = cos(theta[2]);
    float senz = sin(theta[2]);

    cv::Matx<float, 9, 1> R(cosy * cosz, cosx * senz + senx * seny * cosz, senx * senz - cosx * seny * cosz,
        -cosy * senz, cosx * cosz - senx * seny * senz, senx * cosz + cosx * seny * senz,
        seny, -senx * cosy, cosx * cosy);

    return R;
}


vector<Pose> PoseManager::clusterizePoses(const vector<Face> &faces, int numCenters) const
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

vector<vector<const Face*>> PoseManager::assignFacesToClusters(const vector<Face> &faces, const vector<Pose> &centers) const
{
   vector<vector<const Face*>> clusters(centers.size());
   for (const auto &face : faces) {
       int index = getNearestCenterId(face.getRotationMatrix(), centers);
       clusters[index].push_back(&face);
   }
   return clusters;
}

int PoseManager::getNearestCenterId(const Pose &pose, const vector<Pose> &centers) const
{
    float min = FLT_MAX;
    int index = 0;
    for (int i = 0; i < centers.size(); i++) {
        float norm = cv::norm(centers[i].t(), cv::Mat(pose), cv::NORM_L2);
        if (norm < min) {
            min = norm;
            index = i;
        }
    }
    return index;
}

Mat PoseManager::setToCovariance(const vector<const Face*> &set) const
{
    const int SET_SIZE = set.size();
    if (SET_SIZE == 0)
        return Mat::zeros(16,16, CV_32SC1);

    const auto HEIGHT = set[0]->getHeight();
    const auto WIDTH  = set[0]->getWidth();

    const int BLOCK_H = HEIGHT/4;
    const int BLOCK_W  = WIDTH /4;

    vector<vector<Mat>> blocks(SET_SIZE);
    for (int  i = 0; i < SET_SIZE; ++i) {
        for (int x = 0; x < HEIGHT - BLOCK_H; x += BLOCK_H) {
            for (int y = 0; y < WIDTH - BLOCK_W; y += BLOCK_W) {
                cv::Rect roi(x, y, x + BLOCK_W, y + BLOCK_W);
                blocks[i].emplace_back(set[i]->image,roi);
            }
        }
    }

    return Mat::zeros(16,16, CV_32SC1);
}

/*
void PoseManager::addPoseData(const Pose &pose)
{
    posesData.push_back(pose);
}
*/

}   // face
