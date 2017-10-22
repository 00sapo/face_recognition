TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES +=  extern_libs/head_pose_estimation/CRForestEstimator.cpp \
            extern_libs/head_pose_estimation/CRTree.cpp \
            src/covariancecomputer.cpp \
            src/kmeansbackgroundremover.cpp \
            src/svmmanager.cpp \
            src/facecropper.cpp \
            src/main.cpp \
            src/svmstein.cpp \
            src/image4dleaf.cpp \
            src/poseclusterizer.cpp \
            src/svmtester.cpp \
            src/image4dloader.cpp \
            src/preprocessorpipe.cpp \
            src/svmtrainer.cpp \
            src/image4dvectorcomposite.cpp \
            src/settings.cpp \

INCLUDEPATH += include $$(OPENCV_INCLUDE)

HEADERS += include/*.h \
        extern_libs/head_pose_estimation/CRForestEstimator.h \
        extern_libs/head_pose_estimation/CRTree.h \

DISTFILES += \
        camera_info.yaml



unix: LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_ml

unix: LIBS += -lpcl_io -lpcl_common

unix: LIBS += -lboost_system -lpthread -lstdc++fs
