# Overview P2G-ROS suite

Developed and tested for [ROS Melodic](http://wiki.ros.org/melodic) on [Ubuntu 18.04 (Bionic)](http://releases.ubuntu.com/18.04/)


# Setup

### For Ubuntu 18.04 (and maybe other versions)

 *  Follow [this](http://wiki.ros.org/melodic/Installation/Ubuntu) guide to install ROS on your system
 
 *  Install missing dependencies
    
    `sudo apt install python-catkin-tools ros-melodic-rviz-visual-tools ros-melodic-code-coverage`


# Building

 *  Follow Section 3 of [this](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) guide to create your catkin workspace

 *  Clone this repo into the src directory of your catkin workspace
    
    `git@gitlab.com:toposens/ros-projects/ts-ros.git`
    
 * Extract the `ComLib_C_Interface.zip` file from the Position2Go installer into your catkin workspace
    * Your directory should now look similar to this:  
         .  
         ├── p2g_ros  
         ├── p2g_ros_classifier  
         ├── p2g_ros_driver  
         ├──── host_c  
         ├── p2g_ros_msgs  
         ├── p2g_ros_markers  
         └── ...        

    

 *  Build your workspace from inside your catkin folder

    `catkin build p2g_ros_*`


# Running

### Launching the Driver

 *  Make sure you are part of the dialout group:
 
    `sudo adduser $USER dialout`
 
 *  Trigger updated group permissions to take effect:

    `newgrp dialout`

 *  Launch the driver and start accruing data from a TS sensor
 
    `roslaunch p2g_ros_driver p2g_ros_driver.launch`
 
 *  To manipulate sensor parameters live in realtime, run in a new terminal

    `rosrun rqt_reconfigure rqt_reconfigure`
 
---
### Launching the visualization
 
 *  Launch the markers:

    `roslaunch p2g_ros_markers p2g.launch`
    
    * also an instance of the _p2g_ros_driver_ will be launched.

 *  To manipulate markers parameters live in realtime, run in a new terminal:
 
    `rosrun rqt_reconfigure rqt_reconfigure`
---    
### Launching pose classifier node
The pose classifier node subscribes to p2g_ros_msgs.BaseScans which are refactored to be fed into a classifier model.
  *  Launch the classifier:
  
    `roslaunch p2g_ros_classifier p2g_with_pose.launch`  
    
    * also an instance of the _p2g_ros_driver_ will be launched.
  
  * Note: you can also feed the classifier node from a bag file without having a sensor attached, using the following launch file. Please make sure however, that you set a valid path to the rosbag file location in the arguemnts of the launch file.

    `roslaunch p2g_ros_classifier pose_from_bag.launch`


# Using
## Use the P2G_ros_driver in with your packages

To subscribe to the p2g_{base/target}_scan topics, you can take the p2g_ros_pointcloud package as an example.

In general, the following modifications are required:
1. Load the required packages into your catkin workspace
1. Create a build and execution dependency in your `package.xml` to the _p2g_ros_driver_
      ~~~~
     <depend>p2g_ros_driver</depend>
      ~~~~
1. Find and builld the _p2g_ros_driver_ together with your package by adding to your `CMakeLists.txt`
    ~~~~
    find_package(
      catkin REQUIRED 
          ... 
          p2g_ros_driver 
          ...
    )
    ~~~~
1. Launch the _p2g_driver_node_ with your node by adding it to your `*.launch` file
    ~~~~
    <include file="$(find p2g_ros_driver)/launch/p2g_ros_driver.launch"> </include>"
     ~~~~
