#include "posemanager.h"

#include "face.h"
#include "singletonsettings.h"


using std::string;
using std::vector;


const string PoseManager::POSE_ESTIMATOR_PATH = "../trees/";

/*
PoseManager::PoseManager()
    : PoseManager(POSE_ESTIMATOR_PATH)
{
}
*/

PoseManager::PoseManager(const string& poseEstimatorPath)
{
    // load forest for face pose estimation
    if (!estimator.loadForest(poseEstimatorPath.c_str(), 10)) {
        std::cerr << "ERROR! Unable to load forest files" << std::endl;
        return;
    }

    poseEstimatorAvailable = true;
}

bool PoseManager::estimateFacePose(const Face& face, const Mat& calibration)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    SingletonSettings& settings = SingletonSettings::getInstance();
    cv::Mat img3D = face.get3DImage(/*settings.getK()*/calibration);
    //img3D.forEach<cv::Vec3f>([](cv::Vec3f& point, const int* position) {
    //    if (point[2] > 0)
    //        point[2] += 10;
    //});

    cv::imshow("IMage3D", img3D);
    cv::waitKey(0);

    vector<cv::Vec<float, POSE_SIZE>> means; // outputs, POSE_SIZE defined in CRTree.h
    vector<vector<Vote>> clusters; // full clusters of votes
    vector<Vote> votes; // all votes returned by the forest
    int stride = 5;
    float maxVariance = 800;
    float probTH = 1.0;
    float largerRadiusRatio = 1.5;
    float smallerRadiusRatio = 5.0;
    bool verbose = true;
    int threshold = 500;

    estimator.estimate(img3D, means, clusters, votes, stride, maxVariance,
                       probTH, largerRadiusRatio, smallerRadiusRatio,verbose, threshold);

    if (means.empty()) {
        std::cout << "Detection and pose estimation failed!" << std::endl;
        return false;
    }

    for (auto& pose : means) {
        std::cout << "Face detected!" << std::endl;
        std::cout << pose[0] << ", " << pose[1] << ", " << pose[2] << ", "
                  << pose[3] << ", " << pose[4] << ", " << pose[5] << std::endl;
        Vec3f eulerAngles = { pose[0], pose[1], pose[2] };

        posesData.push_back(eulerAnglesToRotationMatrix(eulerAngles));
    }

    return true;
}

Mat PoseManager::eulerAnglesToRotationMatrix(Vec3f& theta)
{
    // Calculate rotation about x axis
    float cosx = cos(theta[0]);
    float senx = sin(theta[0]);
    float cosy = cos(theta[1]);
    float seny = sin(theta[1]);
    float cosz = cos(theta[2]);
    float senz = sin(theta[2]);

    Mat R = (Mat_<float>(9, 1) << cosy * cosz, cosx * senz + senx * seny * cosz, senx * senz - cosx * seny * cosz,
        -cosy * senz, cosx * cosz - senx * seny * senz, senx * cosz + cosx * seny * senz,
        seny, -senx * cosy, cosx * cosy);

    return R;
}

bool PoseManager::clusterizePoses(uint numCenters)
{
    vector<int> bestLabels;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(posesData, numCenters, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.size() != numCenters) {
        std::cout << "Clustering poses failed!" << std::endl;
        return false;
    }

    return true;
}

uint PoseManager::getNearestCenterId(cv::Mat estimation)
{
    float min = FLT_MAX;
    uint index = 0;
    for (uint i = 0; i < centers.size(); i++) {
        float norm = cv::norm(centers.at(i), estimation, NORM_L2);
        if (norm < min) {
            min = norm;
            index = i;
        }
    }
    return index;
}

void PoseManager::addPoseData(cv::Mat pose)
{
    posesData.push_back(pose);
}
