#ifndef SENSOR_H
#define SENSOR_H

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>

#include <p2g_ros_msgs/BaseScans.h>
#include <p2g_ros_msgs/Target.h>
#include <p2g_ros_msgs/TargetScans.h>
#include <p2g_ros_msgs/P2GPoint.h>
#include <p2g_ros_msgs/P2GScan.h>

#include <p2g_ros_driver/p2gConfig.h>

#include <cstdint>
#include <memory.h>

extern "C" {
    #include "EndpointTargetDetection.h"
    #include "EndpointRadarBase.h"
    #include "Protocol.h"
    #include "EndpointRadarAdcxmc.h"
    #include "EndpointRadarP2G.h"
    #include "COMPort.h"
}

namespace p2g_ros_driver
{
/** ROS topic for publishing p2g_scan messages. */
static const char kBaseScansTopic[] = "p2g_base_scans";
static const char kTargetScansTopic[] = "p2g_target_scans";
static const char kPointScansTopic[] = "p2g_point_scans";

/** Maximum number of messages held in buffer for #kScansTopic. */
static const int kQueueSize = 100;

/** @brief  Converts raw sensor data to ROS friendly message structures.
 *  @details  Parses a TsScan from a single input data frame by extracting
 *  its header information and the vector of TsPoints contained in its payload.
 *  A TsScan contains timestamped header information followed by a vector
 *  of TsPoints. A single TsPoint has a 3D location (x, y, z) and an
 *  associated intenstiy. Messages are published to topic #kScansTopic.
 */
class Sensor
{
  public:
    /** Initiates a serial connection and transmits default settings to sensor.
     *  @param nh Public nodehandle for pub-sub ops on ROS topics.
     *  @param private_nh Private nodehandle for accessing launch parameters.
     */
    Sensor(ros::NodeHandle nh, ros::NodeHandle private_nh);
    Sensor(ros::NodeHandle nh, ros::NodeHandle private_nh, std::string port, std::string board_id);

    ~Sensor() {}

    /** Retrieves raw sensor data frames and publishes TsScans extracted from them.
     *  @returns True if scan contains any valid data points. False for an empty scan.
     */
    bool poll(void);

    /** Shuts down serial connection to the sensor. */
    void shutdown(void);

    void received_base_frame_data(void* context,
                                  int32_t protocol_handle,
                                  uint8_t endpoint,
                                  const Frame_Info_t* frame_info
    );

    void received_target_frame_data(void *context, int32_t protocol_handle, uint8_t endpoint, const Target_Info_t *targets, int num_targets);




  private:
    /** Structure generated from cfg file for storing local copy of sensor parameters.*/
    typedef dynamic_reconfigure::Server<p2gConfig> Cfg;

    /** Transmits settings commands on startup with initial data
     *  from the config server.
     */
    void _init(void);

    /** Callback triggered when a parameter is altered on the dynamic
     *  reconfigure server.
     *  Determines which setting has changed and transmits the associated
     *  (well-formed) settings command to the serial stream.
     *  @param cfg Structure holding updated values of all parameters on server.
     *  @param level Indicates parameter that triggered the callback.
     */
    void _reconfigure(p2gConfig &cfg, uint32_t level);


    int radar_auto_connect(void);


    std::string type;
    std::string _frame_id;

    int _radar_handle = 0;
    int _protocolHandle = 0;
    int _endpoint = 0;

    p2gConfig _cfg;    /**< Maintains current values of all config params.*/

    std::unique_ptr<Cfg> _srv;  /**< Pointer to dynamic reconfigure server*/
    ros::Publisher _pub;    /**< Handler for publishing TsScans.*/
    ros::Publisher _point_pub;

    p2g_ros_msgs::BaseScans baseScan;
    p2g_ros_msgs::TargetScans targetScan;
    p2g_ros_msgs::P2GScan p2gScan;

    const double pi = std::acos(-1);

};

} // namespace p2g_ros_driver

#endif
