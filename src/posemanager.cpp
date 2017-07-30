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


bool PoseManager::cropFaces(vector<Image4D>& faces, vector<cv::Rect> &approxFacesRegions)
{
    const auto SIZE = faces.size();
    assert (SIZE == approxFacesRegions.size() &&
            "Every Image4D should have a corresponding approximated face region");

    bool success = true;
    for (uint i = 0; i < SIZE; ++i) {
        success &= cropFace(faces[i], approxFacesRegions[i]);
    }

    return success;
}

bool PoseManager::cropFace(Image4D& face, cv::Rect &approxFaceRegion)
{
    cv::Vec3f eulerAngles;
    if (!estimateFacePose(face, eulerAngles)) {
        return false;
    }

    const std::size_t HEIGHT = face.getHeight();
    const std::size_t WIDTH  = face.getWidth();
    int yTop = 0;
    const int NONZERO_PXL_THRESHOLD = 5;
    for (std::size_t i = 0; i < HEIGHT; ++i) {  // look for first non-empty row
        int nonzeroPixels = 0;
        for (std::size_t j = 0; j < WIDTH; ++j) {
            if (face.depthMap.at<uint16_t>(i,j) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            yTop = i;
            break;
        }
    }

    yTop += 5/8 * eulerAngles[0] + 5/8 * eulerAngles[2];

    int count = 0;
    float meanDist = 0;
    auto lambda = [&] (int x, int y, const uint16_t &depth) {
        if (depth != 0) {
            count++;
            meanDist += float(depth);
        }
    };

    face.depthForEach<uint16_t>(lambda, approxFaceRegion);

    meanDist /= count;
    int yBase = yTop + (130 / (meanDist/1000.f));

    std::cout << "yTop: " << yTop << "\nyBase: " << yBase << std::endl;

    approxFaceRegion = cv::Rect(0,yTop, WIDTH, yBase - yTop);

    int xTop = 0;
    const int MAX_Y = approxFaceRegion.y + approxFaceRegion.height;
    for (std::size_t i = 0; i < WIDTH; ++i) {  // look for first non-empty column
        int nonzeroPixels = 0;
        for (std::size_t j = approxFaceRegion.y; j < MAX_Y; ++j) {
            if (face.depthMap.at<uint16_t>(j,i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xTop = i;
            break;
        }
    }

    int xBase = 0;
    for (std::size_t i = WIDTH; i >= 0; --i) {  // look for last non-empty column
        int nonzeroPixels = 0;
        for (std::size_t j = approxFaceRegion.y; j < MAX_Y; ++j) {
            if (face.depthMap.at<uint16_t>(j,i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xBase = i;
            break;
        }
    }

    approxFaceRegion.x = xTop;
    approxFaceRegion.width = xBase - xTop;

    face.crop(approxFaceRegion);
    return true;
}

bool PoseManager::estimateFacePose(const Image4D& face, cv::Vec3f& eulerAngles)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    cv::Mat img3D = face.get3DImage();
    std::cout << "Image 3d" << std::endl;

    cv::imshow("Image3D", img3D);
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
