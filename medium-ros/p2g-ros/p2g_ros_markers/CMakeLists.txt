cmake_minimum_required(VERSION 2.8.3)
project(p2g_ros_markers)

set(${PROJECT_NAME}_DEPS
  dynamic_reconfigure
  message_runtime
  roscpp
  rospy
  rviz_visual_tools
  p2g_ros_driver
  p2g_ros_msgs
)

find_package(
  catkin REQUIRED
  COMPONENTS ${${PROJECT_NAME}_DEPS}
)


############
## Config ##
############

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${${PROJECT_NAME}_DEPS}
)


###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#############
## Library ##
#############

add_library(
    ${PROJECT_NAME}
    src/lib/plot.cpp
)

add_dependencies(
    ${PROJECT_NAME}
    ${PROJECT_NAME}_gencfg
    ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)


################
## Executable ##
################

add_executable(
    ${PROJECT_NAME}_node
    src/p2g_markers_node.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_node
    ${PROJECT_NAME}
)

install(
    TARGETS ${PROJECT_NAME}_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    COMPONENT main
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)



