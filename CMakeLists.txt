cmake_minimum_required(VERSION 2.8.3)
project(strawberry_detector)

## Compile as C++11, supported in ROS Kinetic and newer
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV 3 REQUIRED)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)