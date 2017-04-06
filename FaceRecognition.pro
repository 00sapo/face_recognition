TEMPLATE = app
CONFIG += console c++14
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

DISTFILES += \
    camera_info.yaml

INCLUDEPATH += $$(OPENCV_INCLUDE)

unix: LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lboost_filesystem -lboost_system
