#ifndef SINGLETONSETTINGS_H
#define SINGLETONSETTINGS_H

#include <opencv2/core.hpp>

using namespace cv;

class SingletonSettings {
public:
    static void setPath(const std::string &pathName);
    static SingletonSettings& getInstance();
    //    ~SingletonSettings();

    const Mat getK();
    const Mat getD();
    const Mat getP();
    const Mat getR();
    int getHeight();
    int getWidth();

    //void setHeight(int height);
    //void setWidth(int width);
    //void setK(Mat K);
    //void setD(Mat D);
    //void setP(Mat P);
    //void setR(Mat R);

    SingletonSettings(SingletonSettings const&) = delete;
    void operator=(SingletonSettings const&)    = delete;

protected:
    SingletonSettings();
    bool read();

    /*
     * Camera parameters
     */
    Mat K;
    Mat D;
    Mat P;
    Mat R;
    int height;
    int width;

    static std::string path;
    static bool pathHasChanged;
};

#endif // SINGLETONSETTINGS_H
