#include "preprocessor.h"

#include <thread>

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

vector<Face> Preprocessor::preprocess(const vector<Image4D> &images)
{
    const auto SIZE = images.size();
    vector<Face> faces(SIZE);

    // get number of concurrently executable threads
    const int numOfThreads = std::thread::hardware_concurrency();
    vector<std::thread> threads(numOfThreads);

    // equally split number of images to load
    int blockSize = SIZE/numOfThreads;
    if (blockSize < 1)
        blockSize = 1;

    std::mutex facesMutex;
    for (int i = 0; i < numOfThreads && i < SIZE; ++i) {
        int begin = i*blockSize;
        int end = begin + blockSize;
        if (i == numOfThreads - 1)
            end += SIZE%numOfThreads;

        // start a thread executing getMultiThr function
        threads[i] = std::thread(&Preprocessor::preprocessMultiThr, this, std::ref(images), std::ref(faces), begin, end, std::ref(facesMutex));
    }

    // wait for threads to end (syncronization)
    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }

    return faces;
}

vector<Image4D> Preprocessor::segment(const std::vector<Image4D> &faces)
{
     vector<Image4D> segmentedImages;
     segmentedImages.reserve(faces.size());

     // for each image...
     for (auto& face : faces) {
         cv::Rect boundingBox;
         // ... detect foreground face...
         if (!detectForegroundFace(face, boundingBox)) {
             std::cout << "No face detected!"
                       << " Applying fixed threshold." << std::endl;
             segmentedImages.push_back(removeBackgroundFixed(face, 1600));
         }
         else {
             segmentedImages.push_back(removeBackgroundDynamic(face, boundingBox));
         }
     }

     return segmentedImages;
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


void Preprocessor::preprocessMultiThr(const vector<Image4D> &images, vector<Face> &faces, int begin, int end, std::mutex &mutex) {
    for (int i = begin; i < end; ++i) {
        auto image = segment(images[i]);
        Vec3f position, eulerAngles;
        cropFace(image, position, eulerAngles);
        std::lock_guard<std::mutex> lock(mutex);
        faces.at(i) = Face(image, position, eulerAngles);
    }
}


Image4D Preprocessor::segment(const Image4D &image4d) {
    cv::Rect boundingBox;
    // ... detect foreground face...
    if (!detectForegroundFace(image4d, boundingBox)) {
        std::cout << "No face detected!"
                  << " Applying fixed threshold." << std::endl;
        return removeBackgroundFixed(image4d, 1600);
    }

    return removeBackgroundDynamic(image4d, boundingBox);
}

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


Image4D Preprocessor::removeBackgroundDynamic(const Image4D& face, const cv::Rect &boundingBox) const
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
        return face;
    }

    // compute threshold based on clustering
    const int FACE_CLUSTER = centers[0] < centers[1] ? 0 : 1;
    float threshold = centers[FACE_CLUSTER] * 1.2f;

    const int MIN_X = boundingBox.x - boundingBox.width;
    const int MAX_X = boundingBox.x + 2*boundingBox.width;

    cv::Mat image;
    face.image.copyTo(image);
    cv::Mat depthMap(face.getHeight(), face.getWidth(), face.depthMap.type());

    face.depthMap.forEach<uint16_t>([&](const uint16_t &p, const int *pos) {
        if ( float(p) > threshold || std::isnan(p) || pos[1] < MIN_X || pos[1] > MAX_X) {
            depthMap.at<uint16_t>(pos[0], pos[1]) = 0;
        }
        else {
            depthMap.at<uint16_t>(pos[0], pos[1]) = p;
        }
    });

    return Image4D(image, depthMap, face.getIntrinsicMatrix());
}

Image4D Preprocessor::removeBackgroundFixed(const Image4D &face, uint16_t threshold) const {

    cv::Mat image;
    face.image.copyTo(image);
    cv::Mat depthMap(face.getHeight(), face.getWidth(), face.depthMap.type());

    // remove background using opencv's parallel foreach to take advantage of multithreading
    auto lambda = [threshold, &depthMap](const uint16_t &p, const int *pos) {
        if (p > threshold || std::isnan(p)) {
            depthMap.at<uint16_t>(pos[0], pos[1]) = 0;
        }
        else {
            depthMap.at<uint16_t>(pos[0], pos[1]) = p;
        }
    };

    face.depthMap.forEach<uint16_t>(lambda);

    return Image4D(image, depthMap, face.getIntrinsicMatrix());
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
