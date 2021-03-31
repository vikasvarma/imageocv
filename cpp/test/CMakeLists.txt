cmake_minimum_required(VERSION 3.0)

# Version Information ----------------------------------------------------------
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

project(
    imageocv_test
    DESCRIPTION "ImageOCV Library Tests"
    LANGUAGES C CXX
)

# Configure OpenCV Dependency --------------------------------------------------
set(OpenCV_DIR 
/Users/vikasvarma/Documents/Development/lib/opencv/install/lib/cmake/opencv4)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR}/include/imageocv )

# Test Executable --------------------------------------------------------------
add_executable( imageocv_test main.cpp)
target_link_libraries( imageocv_test ${OpenCV_LIBS} )
target_link_libraries( imageocv_test ${CMAKE_BINARY_DIR}/libimageocv.a)