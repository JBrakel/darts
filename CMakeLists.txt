cmake_minimum_required(VERSION 3.28)
project(darts)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(OpenVINO_DIR "/opt/intel/openvino_2024.2.0/runtime/cmake")
find_package(OpenVINO REQUIRED)

find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/header)
file(GLOB SOURCES "source/*.cpp")

add_executable(darts main.cpp
               ${SOURCES}
        source/Board.cpp
        header/Board.h
        source/Openvino.cpp
        header/Openvino.h
)

target_link_libraries(darts
        ${OpenCV_LIBS}
        openvino::runtime
        ${X11_LIBRARIES}
)

if(APPLE)
    find_library(COREGRAPHICS_LIBRARY ApplicationServices)
    target_link_libraries(darts ${COREGRAPHICS_LIBRARY})
endif()
