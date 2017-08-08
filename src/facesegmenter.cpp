#include "facesegmenter.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "image4d.h"
#include "singletonsettings.h"

using std::vector;
using std::string;

const string FaceSegmenter::FACE_DETECTOR_PATH = "../haarcascade_frontalface_default.xml";

FaceSegmenter::FaceSegmenter(const string& faceDetectorPath)
{
    // load the pretrained face detection model
    classifier = cv::CascadeClassifier(faceDetectorPath);
    if (classifier.empty()) {
        std::cerr << "ERROR! Unable to load haarcascade_frontalface_default.xml" << std::endl;
        return;
    }

    faceDetectorAvailable = true;
}


 bool FaceSegmenter::segment(vector<Image4D>& faces, vector<cv::Rect> &faceRegions)
 {
     faceRegions.clear();

     bool success = true;
     // for each image...
     for (auto& face : faces) {
         cv::Rect boundingBox;

         // ... detect foreground face...
         if (!detectForegroundFace(face, boundingBox)) {
             std::cout << "No face detected!" << std::endl;
             removeBackgroundFixed(face, 1600);
         }
         else {
             removeBackgroundDynamic(face, boundingBox);
         }

         faceRegions.push_back(boundingBox);
     }

     return success;
 }

bool FaceSegmenter::detectForegroundFace(const Image4D &face, cv::Rect &boundingBox)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(face.image, faces);

    if (faces.empty())
        return false;

    // take face in foregound (the one with bigger bounding box)
    boundingBox = *std::max_element(faces.begin(), faces.end(),
                     [](cv::Rect r1, cv::Rect r2) { return r1.area() < r2.area(); });

    return true;
}


bool FaceSegmenter::removeBackgroundDynamic(Image4D& face, const cv::Rect &boundingBox) const
{
    assert (boundingBox.x > 0 && boundingBox.y > 0
            && boundingBox.x + boundingBox.width <= face.getWidth()
            && boundingBox.y + boundingBox.height <= face.getHeight()
            && "boundingBox must be included in face.image");

    // take non-nan, non-zero points
    vector<float> depth;
    auto lambda = [&depth] (int x, int y, const uint16_t& dpt) {
        if (!std::isnan(dpt) && dpt != 0)
            depth.push_back(dpt);
    };

    face.depthForEach<uint16_t>(lambda, boundingBox);

    // clustering
    vector<int> bestLabels;
    vector<float> centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(depth, 2, bestLabels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);

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

bool FaceSegmenter::removeBackgroundFixed(Image4D& face, uint16_t threshold) const {

    // remove background using opencv's parallel foreach to take advantage of multithreading
    auto lambda = [threshold](uint16_t &p, const int *position) {
        if (p > threshold || p == std::numeric_limits<uint16_t>::quiet_NaN()) {
            p = 0;
        }
    };

    face.depthMap.forEach<uint16_t>(lambda);
}
