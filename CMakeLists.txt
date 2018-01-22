cmake_minimum_required(VERSION 2.8)
project(face_recognition)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_objdetect opencv_ml opencv_calib3d)

set(HEAD_POSE_ESTIM extern_libs/head_pose_estimation)
include_directories(include ${HEAD_POSE_ESTIM})

file (GLOB_RECURSE SRC src/*.cpp)
set (EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${SRC} ${HEAD_POSE_ESTIM}/CRForestEstimator.cpp ${HEAD_POSE_ESTIM}/CRTree.cpp)

target_link_libraries(${EXECUTABLE} ${OpenCV_LIBS} pthread stdc++fs)
