cmake_minimum_required(VERSION 2.8)
project(AgvTracker)

# build flags---------------------------------
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

# find opencv---------------------------------
set(OpenCV_DIR /home/lzh/Downloads/opencv-3.4.16/build)
# set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV 4 REQUIRED)

message(STATUS "found opencv in: " ${OpenCV_INCLUDE_DIRS})

# find Eigen3---------------------------------
find_package(Eigen3 REQUIRED)
message(STATUS "found eigen3 in: " ${EIGEN3_INCLUDE_DIRS})

# build project---------------------------------

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(AgvTracker
    modules/MainApp.cpp
    modules/AgvTracker/AgvTracker.cpp
    modules/Tracking/Tracking.cpp
    modules/Tracking/DegeCorrect.cpp
    modules/Map/Map.cpp
    modules/Frame/Frame.cpp
    modules/Reader/Reader.cpp
    modules/Base/ImgProcess.cpp
    modules/Base/Base.hpp )
    
target_link_libraries(AgvTracker 
			${OpenCV_LIBS} 
			${EIGEN3_LIBS})

