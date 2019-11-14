cmake_minimum_required(VERSION 3.14)

set (CMAKE_CXX_STANDARD 17)

project(imagedescriptors)

set(SOURCE main.cpp)

if (WIN32)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

endif (WIN32)

if (UNIX)

INCLUDE_DIRECTORIES(/usr/local/include/opencv4)
LINK_DIRECTORIES(/usr/local/lib)

endif (UNIX)

add_executable(${PROJECT_NAME} ${SOURCE})


if (WIN32)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

endif (WIN32)

if (UNIX)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} libopencv_core.dylib libopencv_highgui.dylib libopencv_imgcodecs.dylib libopencv_imgproc.dylib)

endif (UNIX)