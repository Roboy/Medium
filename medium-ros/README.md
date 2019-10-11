# Overview Medium-ROS suite

Developed and tested for [ROS Melodic](http://wiki.ros.org/melodic) on [Ubuntu 18.04 (Bionic)](http://releases.ubuntu.com/18.04/)


# Setup

### For Ubuntu 18.04 (and maybe other versions)

 *  Follow [this](http://wiki.ros.org/melodic/Installation/Ubuntu) guide to install ROS on your system
 
 *  Install missing dependencies
    
    `sudo apt install python-catkin-tools ros-melodic-rviz-visual-tools ros-melodic-code-coverage`
    
 * Install Intel RealSense ROS driver following [this](https://github.com/IntelRealSense/realsense-ros) guide


# Building

 *  Follow Section 3 of [this](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) guide to create your catkin workspace

 *  Clone this repo into the src directory of your catkin workspace
    
    `git@gitlab.lrz.de:roboy-medium/medium-ros.git`
    
 *  Build your workspace from inside your catkin folder

    `catkin build medium`


# Running

We have created 3 launch scripts for different purposes, as described in the following:

### Launching a recording session

 *  roslaunch medium record.launch
 
 This launch file will start the real-sense cameras and two sensor handlers, one for multiple toposens TS3 sensors, and one for mulitple P2G sensors.

### Launching a visualization session

 *  roslaunch medium visualize.launch
 
 This launch file will start the sensor handlers, descriptions of the measurement setup and transform as well as visualizes sensor readings in camera frame.
  
### Launching a demo session

 *  roslaunch medium demo.launch
 
 This launch file will start the sensor handlers, descriptions of the measurement setup and camera as well as keypoint visualizer.
  