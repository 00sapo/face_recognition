#include "preprocessor.h"

#include <thread>

#include "face.h"
#include "image4d.h"

using std::string;
using std::vector;
using cv::Vec3f;

namespace face {

const string Preprocessor::FACE_DETECTOR_PATH = "../haarcascade_frontalface_default.xml";
const string Preprocessor::POSE_ESTIMATOR_PATH = "../trees/";

// ---------- constructors ----------

Preprocessor::Preprocessor(const string& faceDetectorPath, const string& poseEstimatorPath)
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

vector<Face> Preprocessor::preprocess(vector<Image4D>& images)
{
    const auto SIZE = images.size();
    vector<Face> croppedFaces;
    croppedFaces.reserve(SIZE);
    int counter = 0;

    for (auto& face : images) {
        uint16_t* th = findDepthThresholds(face);
        cv::Mat copyDepth = face.depthMap.clone();
        removeDepthOutOfThresholds(th[0], th[1], face);

        Vec3f position, eulerAngles;

        bool cropped = false;
        try {
            cropped = cropFace(face, position, eulerAngles);
        } catch (cv::Exception& e) {
            cropped = false;
        }
        while (!cropped && th[0] > 0 && th[1] < UINT16_MAX) {
            std::cout << "cropping failed in " << face.getName() << "! Retrying with new thresholds..." << std::endl;
            face.depthMap = copyDepth.clone();
            th[0] -= 150;
            th[1] -= 150;
            removeDepthOutOfThresholds(th[0], th[1], face);
            try {
                cropped = cropFace(face, position, eulerAngles);
            } catch (cv::Exception& e) {
                cropped = false;
            }
        }

        counter++;
        croppedFaces.emplace_back(face, position, eulerAngles);
    }
    std::cout << "cropped " << counter << " images of " << images.size() << std::endl;

    return croppedFaces;
}

void Preprocessor::findDepthThresholds(std::vector<Image4D>& images)
{
    // for each image...
    for (auto& image : images) {
        findDepthThresholds(image);
    }

    return;
}

// ---------- private member functions ----------

uint16_t* Preprocessor::findDepthThresholds(Image4D& image4d)
{
    // second algorithm

    double max;

    cv::minMaxLoc(image4d.depthMap, NULL, &max, NULL, NULL);

    cv::Mat depthCopy;
    image4d.depthMap.copyTo(depthCopy);

    std::sort(depthCopy.begin<uint16_t>(), depthCopy.end<uint16_t>());

    float count = 1, prevCount = 0, maxFrequency = 0, __maxFrequency__ = 0;
    float prevDiff = 0;
    uint16_t minTh = 0.0, maxTh = 0.0;

    for (uint16_t* p = (uint16_t*)depthCopy.data + 1; p <= (uint16_t*)depthCopy.dataend; p++) {
        uint16_t value = *p;
        if (value >= max / 2 || isnan(value) || value <= 5) {
            continue;
        }

        if (value == *(p - 1)) {
            count++;
        } else {
            count = log10(count);
            float diff = count - prevCount;
            prevCount = count;
            //            if (diff < 5 || diff > -5)
            //                continue;

            if (diff >= 0 && prevDiff <= 0) {
                /* we are in a saddle point (punto di sella) */
                if (__maxFrequency__ > maxFrequency) {
                    maxFrequency = __maxFrequency__;
                    minTh = maxTh;
                    maxTh = value;
                    __maxFrequency__ = 0;
                }
            }

            if (count > __maxFrequency__) {
                __maxFrequency__ = count;
            }
            prevDiff = diff;
            count = 1;
        }
    }
    /* enlarge a bit the thresholds */
    minTh -= 150;
    maxTh += 150;

    uint16_t* th = new uint16_t[2];
    th[0] = minTh;
    th[1] = maxTh;

    return th;
}

void Preprocessor::removeDepthOutOfThresholds(uint16_t minTh, uint16_t maxTh, Image4D& image4d)
{
    image4d.depthMap.forEach<uint16_t>([minTh, maxTh](uint16_t& p, const int* pos) {
        if (p > maxTh || p < minTh || std::isnan(p))
            p = 0;
        return pos;
    });
}

bool Preprocessor::cropFace(Image4D& image4d, Vec3f& position, Vec3f& eulerAngles) const
{
    if (!estimateFacePose(image4d, position, eulerAngles)) {
        return false;
    }

    std::cout << "Face detected!" << std::endl;

    const auto HEIGHT = image4d.getHeight();
    const auto WIDTH = image4d.getWidth();
    const int NONZERO_PXL_THRESHOLD = 5;

    int yTop = 0;
    for (std::size_t i = 0; i < HEIGHT; ++i) { // look for first non-empty row
        int nonzeroPixels = 0;
        for (std::size_t j = 0; j < WIDTH; ++j) {
            if (image4d.depthMap.at<uint16_t>(i, j) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            yTop = i;
            break;
        }
    }

    if (std::abs(eulerAngles[0]) > 35)
        eulerAngles[0] = 0;

    // necessary corrections to take into account head rotations
    yTop += 10 / 8 * eulerAngles[0] + 5 / 8 * eulerAngles[2];
    int yBase = yTop + (145 / (position[2] / 1000.f));
    cv::Rect faceROI(0, yTop, WIDTH, yBase - yTop);

    const int MAX_Y = faceROI.y + faceROI.height - 30; // stay 30px higher to avoid shoulders

    int xTop = 0;
    if (position[1] - 100 < 0 || faceROI.y < 0)
        return false;
    for (int i = position[1] - 100; i < position[1] + 100; ++i) { // look for first non-empty column from left
        int nonzeroPixels = 0;
        for (int j = faceROI.y; j < MAX_Y; ++j) {
            if (image4d.depthMap.at<uint16_t>(j, i) != 0)
                ++nonzeroPixels;
        }
        if (nonzeroPixels >= NONZERO_PXL_THRESHOLD) {
            xTop = i;
            break;
        }
    }

    int xBase = 0;
    if (position[1] - 100 < 0 || faceROI.y < 0)
        return false;
    for (int i = position[1] + 100; i >= position[1] - 100; --i) { // look for last non-empty column from right
        int nonzeroPixels = 0;
        for (int j = faceROI.y; j < MAX_Y; ++j) {

            if (image4d.depthMap.at<uint16_t>(j, i) != 0)
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

bool Preprocessor::estimateFacePose(const Image4D& image4d, cv::Vec3f& position, cv::Vec3f& eulerAngles) const
{
    if (!poseEstimatorAvailable) {
        std::cout << "Error! Face pose estimator unavailable!" << std::endl;
        return false;
    }

    cv::Mat img3D = image4d.get3DImage();

    vector<cv::Vec<float, POSE_SIZE>> means; // outputs, POSE_SIZE defined in CRTree.h
    vector<vector<Vote>> clusters; // full clusters of votes
    vector<Vote> votes; // all votes returned by the forest
    int stride = 10;
    float maxVariance = 800;
    float probTH = 1.0;
    float largerRadiusRatio = 1.5;
    float smallerRadiusRatio = 5.0;
    bool verbose = false;
    int threshold = 13;

    estimator.estimate(img3D, means, clusters, votes, stride, maxVariance,
        probTH, largerRadiusRatio, smallerRadiusRatio, verbose, threshold);

    if (means.empty()) {
        std::cout << "Detection and pose estimation failed!" << std::endl;
        return false;
    }

    auto& pose = means[0];

    position = { -pose[1] + image4d.getHeight() / 2,
        pose[0] + image4d.getWidth() / 2,
        pose[2] };

    eulerAngles = { pose[3], pose[4], pose[5] };

    return true;
}

} // nemaspace face
