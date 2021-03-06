TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += $$files(src/*.cpp) \
		extern_libs/head_pose_estimation/CRForestEstimator.cpp \
		extern_libs/head_pose_estimation/CRTree.cpp

INCLUDEPATH += include $$(OPENCV_INCLUDE) \
		extern_libs/head_pose_estimation/

HEADERS += $$files(include/*.h) \
		CRForestEstimator.h \
		CRTree.h

DISTFILES += \
camera_info.yaml


unix: LIBS += -L$$(OPENCV_LIBS) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_ml -lopencv_calib3d

unix: LIBS += -lpthread -lstdc++fs
