# put output to build dir

cmake_minimum_required(VERSION 3.7)
project(mlp)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(CMAKE_CXX_FLAGS "-std=c++0x -Wno-deprecated")
#set(CMAKE_CXX_STANDARD 11)
#option(NCNN_OPENMP "openmp support" OFF)
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



set(OPENCV_INCLUDE_DIR /data/opencv-3.3.1/include)
set(OPENCV_LIBRARY_DIR /data/opencv-3.3.1/build/lib)
set(OPENCV_LIBS -lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(/usr/include)

add_executable(mlp mlp.cpp)
target_link_libraries(mlp dl ${OpenCV_LIBS} )

