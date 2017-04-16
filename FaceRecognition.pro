TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    singletonsettings.cpp \
    yamlloader.cpp \
    backgroundsegmentation.cpp \
    pointprojector.cpp \
    face.cpp \
    faceloader.cpp

HEADERS += \
    singletonsettings.h \
    yamlloader.h \
    backgroundsegmentation.h \
    pointprojector.h \
    face.h \
    faceloader.h

DISTFILES += \
    camera_info.yaml

INCLUDEPATH += $$(OPENCV_INCLUDE) $$(VTK_INCLUDES)

unix: LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

unix: LIBS += -lboost_filesystem -lboost_system -lboost_iostreams

unix: LIBS += -lpcl_io -lpcl_common -lpcl_visualization -lpcl_filters -lpcl_ml #-lpcl_openni

unix: LIBS += -lvtkCommonDataModel -lvtkRenderingCore -lvtkCommonMath -lvtkCommonCore
