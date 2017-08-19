#include "preprocessor.h"

#include "image4d.h"
#include "face.h"

using std::string;
using std::vector;
using cv::Vec3f;

namespace face {

const string Preprocessor::FACE_DETECTOR_PATH = "../haarcascade_frontalface_default.xml";
const string Preprocessor::POSE_ESTIMATOR_PATH = "../trees/";


// ---------- constructors ----------

Preprocessor::Preprocessor(const string& faceDetectorPath, const std::string &poseEstimatorPath)
{
    // load the pretrained face detection model
    classifier = cv::CascadeClassifier(faceDetectorPath);
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return;
    }

    faceDetectorAvailable = true;

    // load forest for face pose estimation
    if (!estimator.loadForest(poseEstimatorPath.c_str(), 10)) {
        std::cerr << "ERROR! Unable to load forest files" << std::endl;
        return;
    }

    poseEstimatorAvailable = true;
}


// ---------- public member functions ----------

vector<Face> Preprocessor::preprocess(vector<Image4D> &images)
{
    if (!segment(images))
        return vector<Face>();
    return cropFaces(images);
}

bool Preprocessor::segment(vector<Image4D>& faces)
{
     bool success = true;
     // for each image...
     for (auto& face : faces) {
         cv::Rect boundingBox;
         // ... detect foreground face...
         if (!detectForegroundFace(face, boundingBox)) {
             std::cout << "No face detected!"
                       << " Applying fixed threshold." << std::endl;
             success = removeBackgroundFixed(face, 1600);
         }
         else {
             success = removeBackgroundDynamic(face, boundingBox);
         }
     }

     return success;
}

vector<Face> Preprocessor::cropFaces(vector<Image4D> &faces)
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


// ---------- private member functions ----------

bool Preprocessor::detectForegroundFace(const Image4D &face, cv::Rect &boundingBox)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(face.image, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(70, 70));

    if (faces.empty())
        return false;

    // take face in foregound (the one with bigger bounding box)
    boundingBox = *std::max_element(faces.begin(), faces.end(),
                     [](cv::Rect r1, cv::Rect r2) { return r1.area() < r2.area(); });

    return true;
}


bool Preprocessor::removeBackgroundDynamic(Image4D& face, const cv::Rect &boundingBox) const
{
    assert (boundingBox.x > 0 && boundingBox.y > 0
            && boundingBox.x + boundingBox.width <= face.getWidth()
            && boundingBox.y + boundingBox.height <= face.getHeight()
            && "boundingBox must be included in face.image");

    // take non-nan, non-zero points
    vector<float> depth;
    auto lambda = [&depth] (int x, int y, const uint16_t &dpt) {
        if (!std::isnan(dpt) && dpt != 0)
            depth.push_back(dpt);
    };

    face.depthForEach<uint16_t>(lambda, boundingBox);

    // clustering
    vector<int> bestLabels;
    vector<float> centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.size() != 2) {
        std::cout << "Clustering on depth map for background removal failed!" << std::endl;
        return false;
    }

    // compute threshold based on clustering
    const int FACE_CLUSTER = centers[0] < centers[1] ? 0 : 1;
    float threshold = centers[FACE_CLUSTER] * 1.2f;

    const int MIN_X = boundingBox.x - boundingBox.width;
    const int MAX_X = boundingBox.x + 2*boundingBox.width;

    // remove background using opencv's parallel foreach to take advantage of multithreading
    face.depthMap.forEach<uint16_t>([=](uint16_t &p, const int *position) {
        float d = p;
        if (d > threshold || d == std::numeric_limits<float>::quiet_NaN() ||
            position[1] < MIN_X || position[1] > MAX_X) {
            p = 0;
        }
    });

    return true;
}

bool Preprocessor::removeBackgroundFixed(Image4D& face, uint16_t threshold) const {

    // remove background using opencv's parallel foreach to take advantage of multithreading
    auto lambda = [threshold](uint16_t &p, const int *position) {
        if (p > threshold || std::isnan(p)) {
            p = 0;
        }
    };

    face.depthMap.forEach<uint16_t>(lambda);
}

bool Preprocessor::cropFace(Image4D &image4d, Vec3f &position, Vec3f &eulerAngles)
{
    if (!estimateFacePose(image4d, position, eulerAngles)) {
        return false;
    }

    std::cout << "Face detected!" << std::endl;

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

bool Preprocessor::estimateFacePose(const Image4D &image4d, cv::Vec3f &position, cv::Vec3f &eulerAngles)
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

    position    = { -pose[1] + image4d.getHeight() / 2,
                     pose[0] + image4d.getWidth()  / 2,
                     pose[2] };

    eulerAngles = { pose[3], pose[4], pose[5] };

    return true;
}

}   // nemaspace face
