#include "singletonsettings.h"

std::unique_ptr<SingletonSettings> SingletonSettings::instance;

SingletonSettings* SingletonSettings::getInstance()
{
    if(instance == NULL)
        instance = std::unique_ptr<SingletonSettings>(new SingletonSettings());
    return instance.get();
}

Mat SingletonSettings::getK()
{
    return K;
}

Mat SingletonSettings::getD()
{
    return D;
}

Mat SingletonSettings::getP()
{
    return P;
}

Mat SingletonSettings::getR()
{
    return R;
}

int SingletonSettings::getHeight()
{
    return height;
}

int SingletonSettings::getWidth()
{
    return width;
}

void SingletonSettings::setK(Mat K)
{
    this->K = K;
}

void SingletonSettings::setD(Mat D)
{
    this->D = D;
}

void SingletonSettings::setP(Mat P)
{
    this->P = P;
}

void SingletonSettings::setR(Mat R)
{
    this->R = R;
}

void SingletonSettings::setHeight(int height)
{
    this->height = height;
}

void SingletonSettings::setWidth(int width)
{
    this->width = width;
}

SingletonSettings::SingletonSettings()
{
}
