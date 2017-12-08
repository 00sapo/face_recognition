TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += src/main.cpp \
           src/image4d.cpp \
           src/image4dloader.cpp \
           src/face.cpp \
           src/preprocessor.cpp \
           src/settings.cpp \
           src/covariancecomputer.cpp \
           src/facerecognizer.cpp \
           src/svmstein.cpp \
           extern_libs/head_pose_estimation/CRForestEstimator.cpp \
           extern_libs/head_pose_estimation/CRTree.cpp

INCLUDEPATH += include $$(OPENCV_INCLUDE)

HEADERS += include/*.h \
           include/svmstein.h \
           extern_libs/head_pose_estimation/CRForestEstimator.h \
           extern_libs/head_pose_estimation/CRTree.h

DISTFILES += \
    camera_info.yaml



unix: LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_ml

unix: LIBS += -lpthread -lstdc++fs
