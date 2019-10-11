cmake_minimum_required(VERSION 2.8.3)
project(p2g_ros_driver)

set(${PROJECT_NAME}_DEPS
  dynamic_reconfigure
  message_runtime
  roscpp
  rospy
  p2g_ros_msgs
)

set(INFINEON_COM_DIR host_c)

find_package(
  catkin REQUIRED
  COMPONENTS ${${PROJECT_NAME}_DEPS}
)

find_package(
  Boost REQUIRED
  COMPONENTS system
)

############
## Config ##
############
generate_dynamic_reconfigure_options(
  cfg/p2g.cfg
)

catkin_package(
  INCLUDE_DIRS include ${INFINEON_COM_DIR}/include/
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${${PROJECT_NAME}_DEPS}
)

###########
## Build ##
###########
include_directories(
  include/
  host_c/include/
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)


#############
## Library ##
#############
add_library(
  ${PROJECT_NAME}
  src/lib/sensor.cpp
  src/lib/sync_manager.cpp
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

# External Infineon communication library ##############################################################################
add_library(
  INFINEON_LIBRARY
  ${INFINEON_COM_DIR}/src/COMPort_Unix.c
  ${INFINEON_COM_DIR}/src/EndpointCalibration.c
  ${INFINEON_COM_DIR}/src/EndpointRadarAdcxmc.c
  ${INFINEON_COM_DIR}/src/EndpointRadarBase.c
  ${INFINEON_COM_DIR}/src/EndpointRadarDoppler.c
  ${INFINEON_COM_DIR}/src/EndpointRadarErrorCodes.c
  ${INFINEON_COM_DIR}/src/EndpointRadarFmcw.c
  ${INFINEON_COM_DIR}/src/EndpointRadarIndustrial.c
  ${INFINEON_COM_DIR}/src/EndpointRadarP2G.c
  ${INFINEON_COM_DIR}/src/EndpointTargetDetection.c
  ${INFINEON_COM_DIR}/src/Protocol.c
)


################
## Executable ##
################

add_executable(
  ${PROJECT_NAME}_node
  src/p2g_driver_node.cpp
)

target_link_libraries(
  ${PROJECT_NAME}_node
  ${PROJECT_NAME}
  INFINEON_LIBRARY
)

add_executable(
  ${PROJECT_NAME}_sync_node
  src/p2g_sync_node.cpp
)

target_link_libraries(
  ${PROJECT_NAME}_sync_node
  ${PROJECT_NAME}
  INFINEON_LIBRARY
)

###################
## Installations ##
###################

install(
    DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
    DIRECTORY host_c/include
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(
    TARGETS ${PROJECT_NAME}_sync_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    COMPONENT main
)


#install(
#  TARGETS ${PROJECT_NAME}_node
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#  COMPONENT main
#)
#
#install(FILES
#  package.xml
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)
#
#install(DIRECTORY
#  include/p2g_ros_driver/
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#  FILES_MATCHING PATTERN "*.hpp"
#)
#
#install(DIRECTORY
#  ${INFINEON_COM_DIR}/include
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#  FILES_MATCHING PATTERN "*.h"
#)