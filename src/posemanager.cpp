#include "posemanager.h"

#include "image4d.h"
#include "face.h"
#include "singletonsettings.h"


using std::string;
using std::vector;
using cv::Vec3f;

namespace face {


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


vector<Face> PoseManager::cropFaces(vector<Image4D> &faces)
{
    const auto SIZE = faces.size();

    vector<Face> croppedFaces;
    croppedFaces.reserve(SIZE);

    for (auto &face : faces) {
        Vec3f position, eulerAngles;
        cropFace(face, position, eulerAngles);
        croppedFaces.emplace_back(face, position, eulerAngles);
    }

    return croppedFaces;
}

bool PoseManager::cropFace(Image4D &image4d, Vec3f &position, Vec3f &eulerAngles)
{
    imshow("Depth map", image4d.depthMap);
    cv::waitKey(0);

    if (!estimateFacePose(image4d, position, eulerAngles)) {
        return false;
    }

    const std::size_t HEIGHT = image4d.getHeight();
    const std::size_t WIDTH  = image4d.getWidth();
    const int NONZERO_PXL_THRESHOLD = 5;

    int yTop = 0;
    for (std::size_t i = 0; i < HEIGHT; ++i) {  // look for first non-empty row
        int nonzeroPixels = 0;
        for (std::size_t j = 0; j < WIDTH; ++j) {
            if (image4d.depthMap.at<uint16_t>(i,j) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            yTop = i;
            break;
        }
    }

    // necessary corrections to take into account head rotations
    yTop += 10/8 * eulerAngles[0] + 5/8 * eulerAngles[2];
    int yBase = yTop + (145 / (position[2]/1000.f));
    cv::Rect faceROI(0, yTop, WIDTH, yBase - yTop);

    const int MAX_Y = faceROI.y + faceROI.height - 30; // stay 30px higher to avoid shoulders

    int xTop = 0;
    for (int i = position[1] - 100; i < position[1] + 100; ++i) {  // look for first non-empty column from left
        int nonzeroPixels = 0;
        for (int j = faceROI.y; j < MAX_Y; ++j) {
            if (image4d.depthMap.at<uint16_t>(j,i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xTop = i;
            break;
        }
    }

    int xBase = 0;
    for (int i = position[1] + 100; i >= position[1] - 100; --i) {  // look for last non-empty column from right
        int nonzeroPixels = 0;
        for (int j = faceROI.y; j < MAX_Y; ++j) {
            if (image4d.depthMap.at<uint16_t>(j,i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xBase = i;
            break;
        }
    }

    faceROI.x = xTop;
    faceROI.width = xBase - xTop;

    image4d.crop(faceROI);
    return true;
}

bool PoseManager::estimateFacePose(const Image4D &image4d, cv::Vec3f &position, cv::Vec3f &eulerAngles)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    cv::Mat img3D = image4d.get3DImage();

    vector<cv::Vec<float, POSE_SIZE>> means; // outputs, POSE_SIZE defined in CRTree.h
    vector<vector<Vote>> clusters;           // full clusters of votes
    vector<Vote> votes;                      // all votes returned by the forest
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

    position    = { -pose[1] + image4d.getHeight() / 2,
                    pose[0] + image4d.getWidth()  / 2,
                    pose[2] };

    eulerAngles = { pose[3], pose[4], pose[5] };

    std::cout << "Position: " << position[0]
              << ","          << position[1]
              << ","          << position[2] << std::endl;

    std::cout << "Euler angles: " << eulerAngles[0]
              << ","              << eulerAngles[1]
              << ","              << eulerAngles[2] << std::endl;

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

}   // face
