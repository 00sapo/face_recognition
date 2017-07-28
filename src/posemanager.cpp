#include "posemanager.h"

#include "image4d.h"
#include "singletonsettings.h"


using std::string;
using std::vector;


const string PoseManager::POSE_ESTIMATOR_PATH = "../trees/";

PoseManager::PoseManager(const string& poseEstimatorPath)
{
    // load forest for face pose estimation
    if (!estimator.loadForest(poseEstimatorPath.c_str(), 10)) {
        std::cerr << "ERROR! Unable to load forest files" << std::endl;
        return;
    }

    poseEstimatorAvailable = true;
}


bool PoseManager::cropFace(Image4D& face)
{
    cv::Vec3f eulerAngles;
    if (!estimateFacePose(face, eulerAngles)) {
        return false;
    }

    const std::size_t HEIGHT = face.getHeight();
    const std::size_t WIDTH  = face.getWidth();
    int yTop = 0;
    int nonzeroPixels = 0;
    const int NONZERO_PXL_THRESHOLD = 5;
    for (std::size_t i = 0; i < HEIGHT; ++i) {  // look for first non-empty row
        for (std::size_t j = 0; j < WIDTH; ++j) {
            if (float(face.depthMap.at<uint16_t>(i,j)) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            yTop = i;
            break;
        }
    }

    int count = 0;
    float avgDist = 0;
    for (std::size_t i = 0; i < HEIGHT; ++i) {  // compute average distance
        for (std::size_t j = 0; j < WIDTH; ++j) {
            float depth = float(face.depthMap.at<uint16_t>(i,j));
            if (depth > 10E-3f) {
                avgDist += depth;
                ++count;
            }
        }
    }

    avgDist /= count;
    int yBase = yTop + 100 / (avgDist/1000);

    std::cout << "yTop: " << yTop << "\nyBase: " << yBase << std::endl;

    cv::Rect cropRegion(0,yTop, WIDTH, yBase - yTop);
    face.crop(cropRegion);

    imshow("Cropped image", face.image);
    cv::waitKey(0);

    return true;
}

bool PoseManager::estimateFacePose(const Image4D& face, cv::Vec3f& eulerAngles)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    cv::Mat img3D = face.get3DImage();
    std::cout << "IMage 3d" << std::endl;

    cv::imshow("IMage3D", img3D);
    cv::waitKey(0);

    vector<cv::Vec<float, POSE_SIZE>> means; // outputs, POSE_SIZE defined in CRTree.h
    vector<vector<Vote>> clusters; // full clusters of votes
    vector<Vote> votes; // all votes returned by the forest
    int stride = 10;
    float maxVariance = 800;
    float probTH = 1.0;
    float largerRadiusRatio = 1.5;
    float smallerRadiusRatio = 5.0;
    bool verbose = false;
    int threshold = 500;

    estimator.estimate(img3D, means, clusters, votes, stride, maxVariance,
                       probTH, largerRadiusRatio, smallerRadiusRatio,verbose, threshold);

    if (means.empty()) {
        std::cout << "Detection and pose estimation failed!" << std::endl;
        return false;
    }

    auto& pose = means[0];
    std::cout << "Face detected!" << std::endl;
    std::cout << pose[0] << ", " << pose[1] << ", " << pose[2] << ", "
              << pose[3] << ", " << pose[4] << ", " << pose[5] << std::endl;
    eulerAngles = { /*pose[3]*/0, pose[4], pose[5] };

    posesData.push_back(eulerAnglesToRotationMatrix(eulerAngles));

    return true;
}

Pose PoseManager::eulerAnglesToRotationMatrix(cv::Vec3f theta)
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
        float norm = cv::norm(centers.row(i).t(), cv::Mat(poseEstimation), cv::NORM_L2);
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
