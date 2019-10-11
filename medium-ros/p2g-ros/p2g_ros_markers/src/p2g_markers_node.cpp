#include <ros/ros.h>

#include "p2g_ros_markers/plot.h"


int main(int argc, char** argv)
{
  ros::init(argc, argv, "p2g_ros_markers_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  p2g_ros_markers::Plot p(nh, private_nh);
  ros::spin();

  return 0;
}