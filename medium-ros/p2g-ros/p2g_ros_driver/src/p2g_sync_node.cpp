//
// Created by kingkolibri on 29.07.19.
//

#include <ros/ros.h>
#include <p2g_ros_driver/sync_manager.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "multi_p2g_driver_node");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  ros::Rate loop_rate(10); // 10 Hz

  try
  {
    p2g_ros_driver::SensorManager sm(nh, private_nh);

    while (ros::ok())
    {
      sm.trigger();
      ros::spinOnce();
      loop_rate.sleep();
    }

    sm.shutdown();
  }
  catch (const char *msg)
  {
    ROS_ERROR("%s", msg);
  }

  ros::shutdown();
  return 0;
}