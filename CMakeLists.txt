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
  message_generation
)

find_package(OpenCV 3 REQUIRED)

add_service_files(
  FILES
  StwPose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
