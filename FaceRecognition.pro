TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += src/main.cpp \
           extern_libs/head_pose_estimation/CRForestEstimator.cpp \
           extern_libs/head_pose_estimation/CRTree.cpp \
           src/utils.cpp \
           src/steinkernel.cpp \
           src/image4d.cpp \
           src/image4dloader.cpp \
           src/face.cpp \
           src/preprocessor.cpp \
           src/settings.cpp \
    src/covariancecomputer.cpp

HEADERS += \
           extern_libs/head_pose_estimation/CRForestEstimator.h \
           extern_libs/head_pose_estimation/CRTree.h \
           include/test.h \
           include/utils.h \
           include/lbp.h \
           include/steinkernel.h \
           include/svmparams.h \
           include/image4d.h \
           include/image4dloader.h \
           include/face.h \
           include/preprocessor.h \
           include/settings.h \
    include/covariancecomputer.h

DISTFILES += \
    camera_info.yaml

INCLUDEPATH += include $$(OPENCV_INCLUDE) $$(VTK_INCLUDES)

unix: LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect

unix: LIBS += -lboost_filesystem -lboost_system -lboost_iostreams

unix: LIBS += -lpcl_io -lpcl_common -lpcl_visualization -lpcl_filters -lpcl_ml #-lpcl_openni

unix: LIBS += -lvtkCommonDataModel -lvtkRenderingCore -lvtkCommonMath -lvtkCommonCore

unix: LIBS += -lpthread
