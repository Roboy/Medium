cmake_minimum_required(VERSION 2.8.3)
project(medium_localizer)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  LocalizeKeypoints.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
)


catkin_package()



