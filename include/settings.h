#ifndef FACE_SETTINGS_H
#define FACE_SETTINGS_H

#include <opencv2/core.hpp>

namespace face {

class Settings {
public:
    static void setCameraInfoPath(const std::string& pathName);
    static Settings& getInstance();
    //    ~SingletonSettings();

    const cv::Mat getK();
    const cv::Mat getD();
    const cv::Mat getP();
    const cv::Mat getR();
    int getHeight();
    int getWidth();

    //void setHeight(int height);
    //void setWidth(int width);
    //void setK(Mat K);
    //void setD(Mat D);
    //void setP(Mat P);
    //void setR(Mat R);

    Settings(Settings const&) = delete;
    void operator=(Settings const&) = delete;

    std::string getFaceDetectorPath() const;

    std::string getPoseEstimatorPath() const;

protected:
    Settings();
    bool readCameraInfo();

    /*
     * Camera parameters
     */
    cv::Mat K;
    cv::Mat D;
    cv::Mat P;
    cv::Mat R;
    int height;
    int width;

    std::string faceDetectorPath = "../haarcascade_frontalface_default.xml";
    std::string poseEstimatorPath = "../trees/";
    static std::string cameraInfoPath;
    static bool cameraInfoPathHasChanged;
};

} // face

#endif // FACE_SETTINGS_H
