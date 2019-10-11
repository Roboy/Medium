#include "p2g_ros_driver/sync_manager.h"


namespace p2g_ros_driver
{
/** According to the launch parameters, a set of Sensor objects is instantiated.
 *  A dynamic reconfigure server is set up to change the sync mode during runtime.
 */
SensorManager::SensorManager(ros::NodeHandle nh, ros::NodeHandle private_nh)
{
  // Read array of ports and frame IDs for set of sensors from launch parameters
  XmlRpc::XmlRpcValue ports_xmlrpc;
  XmlRpc::XmlRpcValue frames_xmlrpc;
  private_nh.getParam("ports", ports_xmlrpc);
  private_nh.getParam("frames", frames_xmlrpc);

  if (ports_xmlrpc.size() > 4)
  {
    throw std::invalid_argument("Too many ports (>10) given as launch parameters");
  }
  else if (ports_xmlrpc.size() != frames_xmlrpc.size())
  {
    throw std::invalid_argument("Number of ports and frame IDs given as launch parameters not equal");
  }

  // Instantiate sensors
  Sensor * sensor;
  for (int i = 0, n = 0; i < ports_xmlrpc.size(); i++)
  {
    // An indivdual private nodehandle is passed to each Sensor object in order
    // to set the namespace for the dynamic reconfigure of their parameters
    ros::NodeHandle private_nh_sensor("~Sensor" + std::to_string(i));

    try
    {
      ROS_WARN_STREAM(ports_xmlrpc[i] << " " << frames_xmlrpc[i]);
      sensor = new Sensor(nh, private_nh_sensor, ports_xmlrpc[i], frames_xmlrpc[i]);
    }catch(...){
      sensor = nullptr;
      ROS_WARN_STREAM("Sensor " << frames_xmlrpc[i] << " at port " <<  ports_xmlrpc[i] << " failed to be set up!");
      continue;
    }

    _sensors[n++] = sensor;
    _num_sensors = n;
  }

}

/** Triggers all available sensors and polls the received frames from them
 *  in an alternating mode.
 *  @todo enable additional synchronisation modes when real time communication is available
 */
void SensorManager::trigger(void)
{
  for (int i = 0; i < _num_sensors; i++)
  {
    ros::Duration(0.05).sleep();
    _sensors[i]->poll();
  }
}

/** Shuts down serial connections to all sensors. */
void SensorManager::shutdown()
{
  for (int i = 0; i < _num_sensors; i++)
  {
    _sensors[i]->shutdown();
  }
}

}
