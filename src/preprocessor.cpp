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
    segment(images);
    auto faces = cropFaces(images);

    return faces;
}

void Preprocessor::segment(std::vector<Image4D>& images)
{
    // for each image...
    for (auto& image : images) {
        segment(image);
    }

    return;
}

vector<Face> Preprocessor::cropFaces(vector<Image4D>& images)
{
    const auto SIZE = images.size();

    vector<Face> croppedFaces;
    croppedFaces.reserve(SIZE);

    for (auto& face : images) {
        Vec3f position, eulerAngles;
        cropFace(face, position, eulerAngles);
        cv::imshow("face", face.image);
        cv::waitKey(0);
        croppedFaces.emplace_back(face, position, eulerAngles);
    }

    return croppedFaces;
}

// ---------- private member functions ----------

void Preprocessor::segment(Image4D& image4d)
{
    /*
    cv::Rect boundingBox;
    // ... detect foreground face...
    if (!detectForegroundFace(image4d, boundingBox)) {
        std::cout << "No face detected!"
                  << " Applying fixed threshold." << std::endl;
        removeBackgroundFixed(image4d, 1600);
    } else {

        removeBackgroundDynamic(image4d, boundingBox);
    }

   */

    /*
     * This is first reimplementation*/
    double max, min;

    cv::minMaxLoc(image4d.depthMap, &min, &max, NULL, NULL);
    uint16_t MIN = min;
    uint16_t MAX = max;
    int dimension = image4d.getWidth() * image4d.getHeight();
    float fgMin = 0;
    float fgMax = 2.5;
    int countDeleted;
    //    cv::normalize(image4d.depthMap, image4d.depthMap, MIN, MAX, cv::NORM_MINMAX);

    do {
        //        fgMin -= 0.5;
        fgMax += 0.5;
        float threshold1 = (MAX + MIN) / fgMax;
        float threshold2 = 0;

        image4d.depthMap.forEach<uint16_t>([threshold1, threshold2](uint16_t& p, const int* pos) {
            if (p > threshold1 || p < threshold2 || std::isnan(p)) {
                p = 0;
            }
        });
        countDeleted = countNonZero(image4d.depthMap);

    } while ((float)countDeleted / dimension > 0.2);
    /**/

    // second algorithm

    //    double max;

    //    cv::minMaxLoc(image4d.depthMap, NULL, &max, NULL, NULL);

    //    cv::Mat depthCopy = image4d.depthMap.clone();

    //    std::sort(depthCopy.begin<uint16_t>(), depthCopy.end<uint16_t>());

    //    int count = 1, prevCount = 1, maxFrequency = 0, __maxFrequency__ = 0;
    //    int prevDiff = 0;
    //    float minTh = 0.0, maxTh = 0.0, __minTh__ = 0.0, __maxTh__ = 0;

    //    for (uint16_t* p = (uint16_t*)depthCopy.data + 1; p != (uint16_t*)depthCopy.dataend; p++) {
    //        uint16_t value = *p;
    //        if (value >= max / 2 || isnan(value) || value <= 5) {
    //            *p = 0;
    //            continue;
    //        }

    //        if (value == *(p - 1)) {
    //            count++;
    //        } else {

    //            //            int diff = count - prevCount;
    //            //            prevCount = count;
    //            //            if (diff > prevDiff && prevDiff >= 0) {
    //            //                /* histogram is growing a lot */
    //            //                __minTh__ = value;
    //            //            }

    //            //            if (diff < prevDiff && prevDiff <= 0) {
    //            //                /* histogram is decreasing a lot */
    //            //                __maxTh__ = value;
    //            //            }

    //            //            if (diff >= 0 && prevDiff < 0) {
    //            //                /* we are in a saddle point (punto di sella) */
    //            //                if (__maxFrequency__ > maxFrequency) {
    //            //                    maxFrequency = __maxFrequency__;
    //            //                    maxTh = __maxTh__;
    //            //                    minTh = __minTh__;
    //            //                    __maxFrequency__ = 0;
    //            //                }
    //            //            }

    //            if (count > __maxFrequency__) {
    //                __maxFrequency__ = count;
    //                max = value;
    //            }
    //            //            prevDiff = diff;
    //            count = 1;
    //        }
    //    }
    //    //    if (__maxFrequency__ > maxFrequency) {
    //    //        maxTh = __maxTh__;
    //    //        minTh = __minTh__;
    //    //    }

    //    minTh = max - 1000;
    //    maxTh = max + 1000;

    //    image4d.depthMap.forEach<uint16_t>([minTh, maxTh](uint16_t& p, const int* pos) {
    //        if (float(p) > maxTh || float(p) < minTh || std::isnan(p))
    //            p = 0;
    //        return pos;
    //    });

    //    cv::normalize(image4d.depthMap, image4d.depthMap, 0, UINT16_MAX, cv::NORM_MINMAX);
    //    cv::imshow("depth map", image4d.depthMap);
    //    cv::waitKey(0);

    return;
}

bool Preprocessor::detectForegroundFace(const Image4D& face, cv::Rect& boundingBox)
{
    if (!faceDetectorAvailable) {
        std::cout << "Error! Face detector unavailable!" << std::endl;
        return false;
    }

    // face detection
    vector<cv::Rect> faces;
    classifier.detectMultiScale(face.image, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(70, 70));

    if (faces.empty())
        return false;

    // take face in foregound (the one with bigger bounding box)
    boundingBox = *std::max_element(faces.begin(), faces.end(),
        [](cv::Rect r1, cv::Rect r2) { return r1.area() < r2.area(); });

    return true;
}

void Preprocessor::removeBackgroundDynamic(Image4D& face, const cv::Rect& boundingBox) const
{
    assert(boundingBox.x > 0 && boundingBox.y > 0
        && boundingBox.x + boundingBox.width <= face.getWidth()
        && boundingBox.y + boundingBox.height <= face.getHeight()
        && "boundingBox must be included in face.image");

    // take non-nan, non-zero points
    vector<float> depth;
    auto lambda = [&depth](int x, int y, const uint16_t& dpt) {
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
        return;
    }

    // compute threshold based on clustering
    const int FACE_CLUSTER = centers[0] < centers[1] ? 0 : 1;
    float threshold = centers[FACE_CLUSTER] * 1.2f;

    const int MIN_X = boundingBox.x - boundingBox.width;
    const int MAX_X = boundingBox.x + 2 * boundingBox.width;

    face.depthMap.forEach<uint16_t>([&](uint16_t& p, const int* pos) {
        if (float(p) > threshold || std::isnan(p) || pos[1] < MIN_X || pos[1] > MAX_X)
            p = 0;
    });

    return;
}

void Preprocessor::removeBackgroundFixed(Image4D& face, uint16_t threshold) const
{

    face.depthMap.forEach<uint16_t>([threshold](uint16_t& p, const int* pos) {
        if (p > threshold || std::isnan(p))
            p = 0;
    });

    return;
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
