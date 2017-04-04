TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    imageloader.cpp \
    singletonsettings.cpp \
    yamlloader.cpp

HEADERS += \
    imageloader.hpp \
    singletonsettings.h \
    yamlloader.h


unix: LIBS += -lopencv_core -lopencv_videoio -lopencv_imgproc
