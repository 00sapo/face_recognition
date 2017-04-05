#ifndef SINGLETONSETTINGS_H
#define SINGLETONSETTINGS_H

#include <opencv2/core.hpp>

using namespace cv;

class SingletonSettings {
public:
    static SingletonSettings* getInstance();
    //    ~SingletonSettings();

    Mat* getK();
    Mat* getD();
    Mat* getP();
    Mat* getR();
    int getHeight();
    int getWidth();

    void setHeight(int height);
    void setWidth(int width);
    void setK(Mat* K);
    void setD(Mat* D);
    void setP(Mat* P);
    void setR(Mat* R);

private:
    explicit SingletonSettings();
    static SingletonSettings* instance;

    /*
     * Camera parameters
     */
    Mat* K;
    Mat* D;
    Mat* P;
    Mat* R;
    int height;
    int width;
};

#endif // SINGLETONSETTINGS_H
