SET(PROJECT_NAME gpu_svo)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
set(CMAKE_C_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Set definitions
ADD_DEFINITIONS(-DSVO_TRACE)
ADD_DEFINITIONS(-DSVO_USE_ROS)


# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
# Set build flags. Set IS_ARM on odroid board as environment variable
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system)
FIND_PACKAGE(GTSAM REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(vikit_common REQUIRED)

set(GTSAM_INCLUDE_DIRS "${GTSAM_INCLUDE_DIR}")
FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp roslib cmake_modules vikit_common vikit_ros)
catkin_package(
      DEPENDS Eigen OpenCV Sophus Boost fast GTSAM
      CATKIN_DEPENDS roscpp roslib vikit_common vikit_ros
      INCLUDE_DIRS include
      LIBRARIES ${PROJECT_NAME}
)


# Include dirs
INCLUDE_DIRECTORIES(
        include
        kernel
        ${Eigen_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OpenCL_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries
LIST(APPEND LINK_LIBS
  gtsam
  ${OpenCV_LIBS}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${catkin_LIBRARIES}
)


# Set sourcefiles
LIST(APPEND SOURCEFILES
        src/frame_handler_mono.cpp
        src/frame_handler_base.cpp
        src/frame.cpp
        src/point.cpp
        src/map.cpp
        src/pose_optimizer.cpp
        src/initialization.cpp
        src/matcher.cpp
        src/reprojector.cpp
        src/feature_alignment.cpp
        src/feature_detection.cpp
        src/depth_filter.cpp
        src/config.cpp
        src/sparse_img_align.cpp
        src/imu_integration.cpp
        src/cl_class.cpp)

# Create svo library
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCEFILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LINK_LIBS} )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)
ADD_DEFINITIONS(-DKERNEL_DIR=\"${PROJECT_SOURCE_DIR}/kernel\")
