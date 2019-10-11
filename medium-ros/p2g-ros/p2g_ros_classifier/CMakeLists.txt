cmake_minimum_required(VERSION 2.8.3)
project(p2g_ros_classifier)

set(${PROJECT_NAME}_DEPS
  message_runtime
  rospy
  p2g_ros_msgs
)

find_package(
  catkin REQUIRED
  p2g_ros_msgs
  COMPONENTS ${${PROJECT_NAME}_DEPS}
)

find_package(
  Boost REQUIRED
  COMPONENTS system
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS ${${PROJECT_NAME}_DEPS}
)

install()