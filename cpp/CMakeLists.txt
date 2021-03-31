cmake_minimum_required(VERSION 3.0)

# Version Information ----------------------------------------------------------
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set (VERSION_MAJOR 1)
set (VERSION_MINOR 0)
set (VERSION_FEATURE 0)
set (VERSION_PATCH 0)
set (VERSION "${VERSION_MAJOR}.${VERSION_MINOR}")
set (VERSION "${VERSION}.${VERSION_FEATURE}.${VERSION_PATCH}")

project(
    imageocv
    VERSION ${VERSION}
    DESCRIPTION "OpenCV Image Processing & Computer Vision Library"
    LANGUAGES C CXX
)

# Configure OpenCV Dependency --------------------------------------------------
set(OpenCV_DIR 
/Users/vikasvarma/Documents/Development/lib/opencv/install/lib/cmake/opencv4)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Configure IMAGEOCV library ---------------------------------------------------
set(IMAGEOCV_SRC ${CMAKE_SOURCE_DIR}/src)
set(IMAGEOCV_INCLUDE ${CMAKE_SOURCE_DIR}/include/imageocv)

# Header Files:
set(HEADER_FILES
${IMAGEOCV_INCLUDE}/imageio.hpp
${IMAGEOCV_INCLUDE}/findlane.hpp
)

# CXX Source Files:
set(SOURCES
${IMAGEOCV_SRC}/findlane.cpp
${IMAGEOCV_SRC}/imageio.cpp
)

include_directories( ${IMAGEOCV_INCLUDE} )
add_library( imageocv STATIC ${SOURCES} ${HEADER_FILES} )
target_link_libraries( imageocv ${OpenCV_LIBS} )

# Test Executable --------------------------------------------------------------
add_subdirectory(${CMAKE_SOURCE_DIR}/test)