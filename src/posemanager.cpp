#include "posemanager.h"

const string PoseManager::POSE_ESTIMATOR_PATH = "../trees/";

PoseManager::PoseManager()
    : PoseManager(POSE_ESTIMATOR_PATH)
{
}

PoseManager::PoseManager(const std::string& poseEstimatorPath)
{
    // load forest for face pose estimation
    if (!estimator.loadForest(poseEstimatorPath.c_str(), 10)) {
        std::cerr << "ERROR! Unable to load forest files" << std::endl;
        return;
    }

    poseEstimatorAvailable = true;
}

bool PoseManager::estimateFacePose(const Face& face)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    SingletonSettings& settings = SingletonSettings::getInstance();
    cv::Mat img3D = face.get3DImage(settings.getK());

    cv::imshow("IMage3D", img3D);
    cv::waitKey(0);

    vector<cv::Vec<float, POSE_SIZE>> means; // outputs
    vector<vector<Vote>> clusters; // full clusters of votes
    vector<Vote> votes; // all votes returned by the forest
    int stride = 5;

    estimator.estimate(img3D, means, clusters, votes, stride, 800);

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

Pose PoseManager::eulerAnglesToRotationMatrix(Vec3f theta)
{
    // Calculate rotation about x axis
    float cosx = cos(theta[0]);
    float senx = sin(theta[0]);
    float cosy = cos(theta[1]);
    float seny = sin(theta[1]);
    float cosz = cos(theta[2]);
    float senz = sin(theta[2]);

    Matx<float, 9, 1> R(cosy * cosz, cosx * senz + senx * seny * cosz, senx * senz - cosx * seny * cosz,
        -cosy * senz, cosx * cosz - senx * seny * senz, senx * cosz + cosx * seny * senz,
        seny, -senx * cosy, cosx * cosy);

    return R;
}

bool PoseManager::clusterizePoses(int numCenters)
{
    vector<int> bestLabels;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(posesData, numCenters, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.rows != numCenters) {
        std::cout << "Clustering poses failed!" << std::endl;
        return false;
    }

    return true;
}

int PoseManager::getNearestCenterId(Pose poseEstimation)
{
    float min = FLT_MAX;
    int index = 0;
    for (int i = 0; i < centers.rows; i++) {
        float norm = cv::norm(centers.row(i).t(), (Mat)poseEstimation, NORM_L2);
        if (norm < min) {
            min = norm;
            index = i;
        }
    }
    return index;
}

void PoseManager::addPoseData(Pose pose)
{
    posesData.push_back(pose);
}
