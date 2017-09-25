#include "preprocessor.h"

#include <thread>

#include "face.h"
#include "image4d.h"

using cv::Vec3f;
using std::string;
using std::vector;

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
    const ulong SIZE = images.size();
    vector<Face> croppedFaces;
    croppedFaces.reserve(SIZE);
    int counter = 0;

    std::cout << "Preprocessing... " << std::endl;
    for (auto& face : images) {
        vector<threshold> thresholds;
        findDepthThresholds(face, thresholds, 4);
        for (auto th : thresholds) {
            Vec3f position, eulerAngles;

            cv::Mat copyDepth = face.depthMap.clone();

            bool cropped = filterAndCrop(face, th.minTh, th.maxTh, position, eulerAngles);

            if (cropped) {
                //                std::cout << face.getName() << " cropped!" << std::endl;
                counter++;
                croppedFaces.emplace_back(face, position, eulerAngles);
                break;
            } else {
                cv::imshow(face.getName(), face.depthMap);
                cv::waitKey(0);
                //                std::cout << "cropping failed in " << face.getName() << "! Retrying with new thresholds..." << std::endl;
                face.depthMap = copyDepth.clone();
            }
            croppedFaces.emplace_back(face, position, eulerAngles);
        }
    }
    printf("\n");
    std::cout << "cropped " << counter << " images of " << images.size() << std::endl;

    return croppedFaces;
}

bool Preprocessor::filterAndCrop(Image4D& face, uint16_t minTh, uint16_t maxTh, Vec3f& position, Vec3f& eulerAngles)
{

    face.depthMap.forEach<uint16_t>([minTh, maxTh](uint16_t& p, const int* pos) {
        if (p > maxTh || p < minTh || std::isnan(p))
            p = 0;
    });
    //    cv::imshow(face.getName(), face.depthMap);
    //    cv::waitKey(0);

    bool cropped = false;
    try {
        cropped = cropFace(face, position, eulerAngles);
    } catch (cv::Exception& e) {
        cropped = false;
    }

    return cropped;
}

/*void Preprocessor::findDepthThresholds(std::vector<Image4D>& images)
{
    // for each image...
    for (auto& image : images) {
        findDepthThresholds(image);
    }

    return;
}*/

// ---------- private member functions ----------
int Preprocessor::findLeastFreqTh(std::vector<threshold> interestingThs)
{
    float min = FLT_MAX;
    vector<threshold>::iterator minIt;
    for (vector<threshold>::iterator it = interestingThs.begin(); it < interestingThs.end(); it++) {
        threshold v = *it;
        if (v.freq < min) {
            min = v.freq;
            minIt = it;
        }
    }
    return minIt - interestingThs.begin();
}

void Preprocessor::findDepthThresholds(Image4D& image4d, std::vector<threshold>& interestingThs, int k)
{
    // second algorithm

    double max;

    cv::minMaxLoc(image4d.depthMap, NULL, &max, NULL, NULL);

    cv::Mat depthCopy;
    image4d.depthMap.copyTo(depthCopy);

    std::sort(depthCopy.begin<uint16_t>(), depthCopy.end<uint16_t>());

    float prevFreq = 0, minInterestingFreq = 0;
    float freq = 0, localMaxFreq = 0;
    float prevDiff = 0;
    threshold newTh = { 0, 0, 0 };
    //    uint16_t minTh = 0, temp = 0;

    for (uint16_t* p = (uint16_t*)depthCopy.data + 1; p <= (uint16_t*)depthCopy.dataend; p++) {
        uint16_t value = *p;
        if (isnan(value) || value <= 5 || value <= newTh.maxTh) {
            /* filtering everything useless */
            continue;
        }
        if (value >= max / 2)
            break;

        if (value == *(p - 1)) {
            freq++;
        } else {
            /* since depthCopy has been ordered, we have just counted the number of equal values */
            /* logarithm decreases local maximum, derivative finds maximums and saddle points */
            freq = log10(freq);
            float diff = freq - prevFreq;
            prevFreq = freq;

            if (freq > localMaxFreq)
                localMaxFreq = freq;

            if (diff >= 0 && prevDiff <= 0) {
                /* we are in a saddle point (punto di sella) */
                /* memorizing new threshold found */
                newTh.freq = localMaxFreq;
                newTh.minTh = newTh.maxTh;
                newTh.maxTh = value + 300;
                //                if (newTh.minTh != 0) {
                //                temp = newTh.maxTh;
                if (interestingThs.size() < k)
                    interestingThs.push_back(newTh);
                else if (localMaxFreq >= minInterestingFreq) {
                    //find least frequent value
                    int pos = findLeastFreqTh(interestingThs);
                    // updating it
                    interestingThs.at(pos) = newTh;
                    // updating minimum interesting frequency at now
                    pos = findLeastFreqTh(interestingThs);
                    minInterestingFreq = interestingThs.at(pos).freq;
                    localMaxFreq = 0;
                    //                    }
                }
            }

            prevDiff = diff;
            freq = 1;
        }

        //sorting largest to smallest
        std::sort(interestingThs.begin(), interestingThs.end(),
            [](const threshold& a, const threshold& b) {
                return a.maxTh - a.minTh < b.maxTh - b.minTh;
            });
    }
}

bool Preprocessor::cropFace(Image4D& image4d, Vec3f& position, Vec3f& eulerAngles) const
{
    if (!estimateFacePose(image4d, position, eulerAngles)) {
        return false;
    }

    //    std::cout << "Face detected in " << image4d.getName() << std::endl;

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

    if (faceROI.x < 0 || faceROI.y < 0 || faceROI.area() <= 0)
        return false;

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
