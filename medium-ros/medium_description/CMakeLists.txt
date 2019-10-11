cmake_minimum_required(VERSION 2.8.3)
project(medium_description)

add_compile_options(-std=c++17)

set(${PROJECT_NAME}_DEPS
	urdf
	xacro
)

find_package(
	catkin REQUIRED
	COMPONENTS ${${PROJECT_NAME}_DEPS}
)

catkin_package(
	CATKIN_DEPENDS ${${PROJECT_NAME}_DEPS}
)

install(
  DIRECTORY launch meshes rviz urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
