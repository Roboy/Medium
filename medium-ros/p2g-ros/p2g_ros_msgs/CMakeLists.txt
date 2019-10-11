cmake_minimum_required(VERSION 2.8.3)
project(p2g_ros_msgs)

find_package(
	catkin REQUIRED
	COMPONENTS
	geometry_msgs
	message_generation
	std_msgs
)

add_message_files(
	DIRECTORY msg
	FILES
	Target.msg
	TargetScans.msg
	Sample.msg
	Antenna.msg
	Chirp.msg
	BaseScans.msg
	P2GPoint.msg
	P2GScan.msg
)

generate_messages(
	DEPENDENCIES
	std_msgs
	geometry_msgs
)

catkin_package(
	CATKIN_DEPENDS
	message_runtime
	std_msgs
)