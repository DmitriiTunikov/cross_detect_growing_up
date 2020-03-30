cmake_minimum_required(VERSION 2.8)

project( CrossDetector )

find_package( OpenCV REQUIRED )

add_executable( ${PROJECT_NAME} main.cpp cross_detector.h cross_detector.cpp cv_supp.h cv_supp.cpp draw.cpp)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )