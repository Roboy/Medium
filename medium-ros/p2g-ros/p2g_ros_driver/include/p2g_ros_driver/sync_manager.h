#ifndef SENSOR_MANAGER_H
#define SENSOR_MANAGER_H

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <ros/xmlrpc_manager.h>

#include "p2g_ros_driver/sensor.h"


namespace p2g_ros_driver
{

/** @brief  Manages a system of multiple TS sensors.
 *  @detail  Instantiates a set of TS sensors. Tells the sensors to scan in an
 *  alternating mode.
 */
class SensorManager
{
  public:
    /** Instantiates a set of Sensor objects according to the launch parameters.
     *  @param nh Public nodehandle for pub-sub ops on ROS topics.
     *  @param private_nh Private nodehandle for accessing launch parameters.
    */
    SensorManager(ros::NodeHandle nh, ros::NodeHandle private_nh);
    ~SensorManager() {this->shutdown();}

    /** Triggers all available sensors and polls the received frames from them
     *  in an alternating mode.
     */
    void trigger(void);

    /** Shuts down serial connections to all sensors. */
    void shutdown(void);

  private:
    Sensor *_sensors[4];  /**< Array of pointers to sensors (a maximum number of 4 sensors can be used).*/
    int _num_sensors;  /**< Number of sensors that are used.*/
};

} // namespace toposens_driver

#endif
