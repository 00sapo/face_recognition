TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += src/main.cpp \
           src/image4dloader.cpp \
           src/settings.cpp \
           src/covariancecomputer.cpp \
           src/svmstein.cpp \
           extern_libs/head_pose_estimation/CRForestEstimator.cpp \
           extern_libs/head_pose_estimation/CRTree.cpp \
    svmtester.cpp \
    src/svmtrainer.cpp \
    svmmanager.cpp \
    kmeansbackgroundremover.cpp \
    facecropper.cpp \
    src/image4dleaf.cpp \
    image4dclustercomposite.cpp \
    preprocessorpipe.cpp \
    poseclusterizer.cpp

INCLUDEPATH += include $$(OPENCV_INCLUDE)

HEADERS += include/*.h \
           include/svmstein.h \
           extern_libs/head_pose_estimation/CRForestEstimator.h \
           extern_libs/head_pose_estimation/CRTree.h \
    svmtester.h \
    svmmanager.h \
    filter.h \
    kmeansbackgroundremover.h \
    facecropper.h \
    image4dsetcomponent.h \
    image4dclustercomposite.h \
    preprocessorpipe.h \
    poseclusterizer.h

DISTFILES += \
    camera_info.yaml



unix: LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_ml

unix: LIBS += -lpcl_io -lpcl_common

unix: LIBS += -lboost_system -lpthread -lstdc++fs
