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

bool PoseManager::cropFace(Image4D& face, cv::Rect &faceROI)
{
    cv::Vec3f position, eulerAngles;

    imshow("Depth map", face.depthMap);
    cv::waitKey(0);

    if (!estimateFacePose(face, position, eulerAngles)) {
        return false;
    }

    removeOutlierBlobs(face, position);

    imshow("Outlier removed", face.depthMap);
    cv::waitKey(0);

    const std::size_t HEIGHT = face.getHeight();
    const std::size_t WIDTH  = face.getWidth();
    const int NONZERO_PXL_THRESHOLD = 5;

    int yTop = 0;
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

    // necessary corrections to take into account head rotations
    yTop += 10/8 * eulerAngles[0] + 5/8 * eulerAngles[2];
    int yBase = yTop + (145 / (position[2]/1000.f));
    faceROI = cv::Rect(0, yTop, WIDTH, yBase - yTop);

    const int MAX_Y = faceROI.y + faceROI.height - 30; // stay 30px higher to avoid shoulders

    int xTop = 0;
    for (std::size_t i = 0; i < WIDTH; ++i) {  // look for first non-empty column
        int nonzeroPixels = 0;
        for (std::size_t j = faceROI.y; j < MAX_Y; ++j) {
            if (face.depthMap.at<uint16_t>(j,i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xTop = i;
            break;
        }
    }

    int xBase = 0;
    for (std::size_t i = WIDTH-1; i >= 0; --i) {  // look for last non-empty column
        int nonzeroPixels = 0;
        for (std::size_t j = faceROI.y; j < MAX_Y; ++j) {
            if (face.depthMap.at<uint16_t>(j,i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xBase = i;
            break;
        }
    }

    faceROI.x = xTop;
    faceROI.width = xBase - xTop;

    face.crop(faceROI);
    return true;
}

bool PoseManager::estimateFacePose(const Image4D &face, cv::Vec3f &position, cv::Vec3f &eulerAngles)
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    std::ofstream file;
    file.open("/home/alberto/Desktop/depthMap_multi.txt");
    for (int i = 0; i < face.depthMap.rows; ++i) {
        for (int j = 0; j < face.depthMap.cols; ++j) {
            file << face.depthMap.at<uint16_t>(i,j) << " ";
        }
        file << std::endl;
    }
    file.close();

    cv::Mat img3D = face.get3DImage();

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

    file.open("/home/alberto/Desktop/img3D_multi.txt");
    for (int i = 0; i < img3D.rows; ++i) {
        for (int j = 0; j < img3D.cols; ++j) {
            const auto& vec = img3D.at<cv::Vec3f>(i,j);
            file << vec[0] << " " << vec[1] << " " << vec[2] << " ";
        }
        file << std::endl;
    }
    file.close();

    estimator.estimate(img3D, means, clusters, votes, stride, maxVariance,
                       probTH, largerRadiusRatio, smallerRadiusRatio,verbose, threshold);

    if (means.empty()) {
        std::cout << "Detection and pose estimation failed!" << std::endl;
        return false;
    }

    auto& pose = means[0];
    std::cout << "Face detected!" << std::endl;

    position    = { -pose[1] + face.getHeight() / 2,
                    pose[0] + face.getWidth()  / 2,
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


void PoseManager::removeOutlierBlobs(Image4D &face, const cv::Vec3f &position) const {

//  int minX = position[0] - 100;
//  int maxX = position[0] + 100;
    int minY = position[1] - 100;
    int maxY = position[1] + 100;

    face.depthMap.forEach<uint16_t>([=](uint16_t &p, const int* pos) {
        if (/*pos[0] < minX || pos[0] > maxX ||*/ pos[1] < minY || pos[1] > maxY) {
            p = 0;
        }
    });

/*
    face.depthMap.forEach<uint16_t>([=](uint16_t &p, const int* pos) {
        if (!(pos[0] < minX || pos[0] > maxX || pos[1] < minY || pos[1] > maxY)) {
            p = 10000;
        }
    });
*/

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
