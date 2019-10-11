#include <p2g_ros_driver/sensor.h>

class p2g_target;
namespace p2g_ros_driver {


/**
 * Template bounce function to redirect C-callback to C++ class member method with reference priv.
 */
    template<class T, class Method, Method m, class Ret, class ...Args>
    static Ret bounce(void *priv, Args... args) {
      return ((*reinterpret_cast<T *>(priv)).*m)(NULL, args...);
    }
/**
* Convenience macro to simplify bounce statement usage
*/
#define BOUNCE(c, m) bounce<c, decltype(&c::m), &c::m>


/** A dynamic reconfigure server is set up to configure sensor
 *  performance parameters during runtime.
 */
    Sensor::Sensor(ros::NodeHandle nh, ros::NodeHandle private_nh) {
      private_nh.getParam("frame_id", _frame_id);
      private_nh.getParam("type", type);

      // Set up connection to COM port -------------------------------------------------------------------------------
      int res = -1;

      // open COM port
      _protocolHandle = radar_auto_connect();

      // get endpoint ids
      if (_protocolHandle >= 0) {
        for (int i = 1; i <= protocol_get_num_endpoints(_protocolHandle); ++i) {
          // current endpoint is radar base endpoint

          if (type == "target") {
            if (ep_targetdetect_is_compatible_endpoint(_protocolHandle, i) == 0) {
              _endpoint = i;
              if (_endpoint >= 0) {
                ep_targetdetect_set_callback_target_processing(&BOUNCE(Sensor, received_target_frame_data),
                                                               this);
                res = 1;
              }
              continue;
            }
          } else { // type == 'base'
            if (ep_radar_base_is_compatible_endpoint(_protocolHandle, i) == 0) {
              _endpoint = i;
              if (_endpoint >= 0) {
                ep_radar_base_set_callback_data_frame(&BOUNCE(Sensor, received_base_frame_data), this);
                res = ep_radar_base_set_automatic_frame_trigger(_protocolHandle, _endpoint, 0);
              }
              continue;
            }
          }
        }
      }


      if (res == -1) {
        ROS_ERROR("Failed to establish connection to radar sensor! Aborting!");
        return;
      }

      // Set up dynamic reconfigure to change sensor settings --------------------------------------------------------
      _srv = std::make_unique<Cfg>(private_nh);
      Cfg::CallbackType f = boost::bind(&Sensor::_reconfigure, this, _1, _2);
      _srv->setCallback(f);

      // Publishing topic for TsScans --------------------------------------------------------------------------------
      if (type == "target") {
        _pub = nh.advertise<p2g_ros_msgs::TargetScans>(kTargetScansTopic, kQueueSize);
        _point_pub = nh.advertise<p2g_ros_msgs::P2GScan>(kPointScansTopic, kQueueSize);
        ROS_INFO("Publishing Position2Go samples to /%s", kTargetScansTopic);
      } else { // type == "base"
        _pub = nh.advertise<p2g_ros_msgs::BaseScans>(kBaseScansTopic, kQueueSize);
        ROS_INFO("Publishing Position2Go samples to /%s", kBaseScansTopic);
      }

    }

    Sensor::Sensor(ros::NodeHandle nh, ros::NodeHandle private_nh, std::string port, std::string frame_id) {
      private_nh.getParam("type", type);
      _frame_id = frame_id;

      // Set up connection to COM port -------------------------------------------------------------------------------
      int res = -1;

      // open COM port
      _protocolHandle = protocol_connect(port.c_str());

      // get endpoint ids
      if (_protocolHandle >= 0) {
        for (int i = 1; i <= protocol_get_num_endpoints(_protocolHandle); ++i) {
          // current endpoint is radar base endpoint

          if (type == "target") {
            if (ep_targetdetect_is_compatible_endpoint(_protocolHandle, i) == 0) {
              _endpoint = i;
              if (_endpoint >= 0) {
                ep_targetdetect_set_callback_target_processing(&BOUNCE(Sensor, received_target_frame_data),
                                                               this);
                res = 1;
              }
              continue;
            }
          } else { // type == 'base'
            if (ep_radar_base_is_compatible_endpoint(_protocolHandle, i) == 0) {
              _endpoint = i;
              if (_endpoint >= 0) {
                ep_radar_base_set_callback_data_frame(&BOUNCE(Sensor, received_base_frame_data), this);
                res = ep_radar_base_set_automatic_frame_trigger(_protocolHandle, _endpoint, 0);

              }
              continue;
            }
          }

        }
      }


      if (res == -1) {
        ROS_ERROR("Failed to establish connection to radar sensor! Aborting!");
        return;
      }

      // Set up dynamic reconfigure to change sensor settings --------------------------------------------------------
      _srv = std::make_unique<Cfg>(private_nh);
      Cfg::CallbackType f = boost::bind(&Sensor::_reconfigure, this, _1, _2);
      _srv->setCallback(f);

      // Publishing topic for TsScans --------------------------------------------------------------------------------
      if (type == "target") {
        _pub = nh.advertise<p2g_ros_msgs::TargetScans>(kTargetScansTopic, kQueueSize);
        _point_pub = nh.advertise<p2g_ros_msgs::P2GScan>(kPointScansTopic, kQueueSize);
        ROS_INFO("Publishing Position2Go samples to /%s", kTargetScansTopic);
      } else { // type == "base"
        _pub = nh.advertise<p2g_ros_msgs::BaseScans>(kBaseScansTopic, kQueueSize);
        ROS_INFO("Publishing Position2Go samples to /%s", kBaseScansTopic);
      }

    }

    /**
     * Automatically query ports to detect connected single P2G board endpoints.
     * @return
     */
    int Sensor::radar_auto_connect(void) {
      int num_of_ports = 0;
      char comp_port_list[256];
      char *comport;
      const char *delim = ";";

      num_of_ports = com_get_port_list(comp_port_list, (size_t) 256);

      if (num_of_ports == 0) return -1;
      else {
        comport = strtok(comp_port_list, delim);

        while (num_of_ports > 0) {
          num_of_ports--;

          // open COM port
          _radar_handle = protocol_connect(comport);

          if (_radar_handle >= 0) {
            ROS_INFO("Device at %s ready for communication", comport);
            break;
          }

          comport = strtok(nullptr, delim);
        }

        return _radar_handle;
      }

    }

    /** Parse radar base frames.
     *  Registered as callback to ep_radar_base_get_frame_data method to return measured time domain signals.
     */
    void Sensor::received_base_frame_data(void *context,
                                          int32_t protocol_handle,
                                          uint8_t endpoint,
                                          const Frame_Info_t *frame_info
    ) {

      std::ostringstream stringStream;
      stringStream << "p2g_" << protocol_handle;

      baseScan.header.stamp = ros::Time::now();
      baseScan.header.frame_id = stringStream.str();
      baseScan.chirps.clear();

      int num_chirps = frame_info->num_chirps;
      int num_samples_per_chirp = frame_info->num_samples_per_chirp;
      int num_rx_antennas = frame_info->num_rx_antennas;
      int data_format = frame_info->data_format;

      ROS_DEBUG_STREAM("num_chirps " << num_chirps << std::endl <<
                                     "num_samples_per_chirp " << num_samples_per_chirp << std::endl <<
                                     "num_rx_antennas " << num_rx_antennas << std::endl <<
                                     "data_format " << data_format << std::endl
      );


      for (size_t chirp = 0; chirp < frame_info->num_chirps; chirp++) {
        auto frame_start = &frame_info->sample_data[chirp *
                                                    num_rx_antennas *
                                                    num_samples_per_chirp *
                                                    ((data_format == EP_RADAR_BASE_RX_DATA_REAL) ? 1 : 2)];

        p2g_ros_msgs::Chirp chirp_msgs;

        for (uint8_t antenna = 0; antenna < num_rx_antennas; antenna++) {
          p2g_ros_msgs::Antenna antenna_msg;

          for (uint32_t sample = 0; sample < frame_info->num_samples_per_chirp; sample++) {
            p2g_ros_msgs::Sample sample_msg;

            if (frame_info->interleaved_rx == 0 && frame_info->data_format == EP_RADAR_BASE_RX_DATA_REAL) {
              sample_msg.real = frame_start[antenna * num_samples_per_chirp + sample];
              sample_msg.imag = 0;

            } else if (frame_info->interleaved_rx == 0 && frame_info->data_format == EP_RADAR_BASE_RX_DATA_COMPLEX) {
              sample_msg.real = frame_start[(2 * antenna) * num_samples_per_chirp + sample];
              sample_msg.imag = frame_start[(2 * antenna + 1) * num_samples_per_chirp + sample];

            } else if (frame_info->interleaved_rx == 0 &&
                       frame_info->data_format == EP_RADAR_BASE_RX_DATA_COMPLEX_INTERLEAVED) {
              sample_msg.real = frame_start[2 * antenna * num_samples_per_chirp + sample];
              sample_msg.imag = frame_start[2 * antenna * num_samples_per_chirp + sample + 1];

            } else if (frame_info->interleaved_rx == 1 && frame_info->data_format == EP_RADAR_BASE_RX_DATA_REAL) {
              sample_msg.real = frame_start[sample * num_rx_antennas + antenna];
              sample_msg.imag = 0;

            } else if (frame_info->interleaved_rx == 1 && frame_info->data_format == EP_RADAR_BASE_RX_DATA_COMPLEX) {
              sample_msg.real = frame_start[sample * num_rx_antennas + antenna];
              sample_msg.imag = frame_start[(num_samples_per_chirp + sample) * num_rx_antennas + antenna];

            } else if (frame_info->interleaved_rx == 1 &&
                       frame_info->data_format == EP_RADAR_BASE_RX_DATA_COMPLEX_INTERLEAVED) {
              sample_msg.real = frame_start[2 * sample * num_rx_antennas + antenna];
              sample_msg.imag = frame_start[2 * sample * num_rx_antennas + antenna + 1];

            } else {
              ROS_WARN("Undefined RadarBaseFrame format.");
              continue;
            }

            antenna_msg.samples.push_back(sample_msg);
          }

          chirp_msgs.antennas.push_back(antenna_msg);

        }

        baseScan.chirps.push_back(chirp_msgs);
      }

      _pub.publish(baseScan);

    }

    /** Parse radar base frames.
     *  Registered as callback to ep_radar_base_get_frame_data method to return measured time domain signals.
     */
    void Sensor::received_target_frame_data(void *context, int32_t protocol_handle, uint8_t endpoint,
                                            const Target_Info_t *targets, int num_targets) {

      for (uint32_t i = 0; i < num_targets; i++) {
        p2g_ros_msgs::Target target;

        target.target_id = targets[i].target_id;
        target.level = targets[i].level;
        target.radius = targets[i].radius;
        target.azimuth = targets[i].azimuth;
        target.elevation = targets[i].elevation;
        target.radial_speed = targets[i].radial_speed;
        target.azimuth_speed = targets[i].azimuth_speed;
        target.elevation_speed = targets[i].elevation_speed;

        targetScan.targets.push_back(target);

        p2g_ros_msgs::P2GPoint point;
        point.location.x =
                target.radius / 100 * std::cos(target.elevation * pi / 180) * std::cos(target.azimuth * pi / 180);
        point.location.y =
                target.radius / 100 * std::cos(target.elevation * pi / 180) * std::sin(target.azimuth * pi / 180);
        point.location.z = target.radius / 100 * std::sin(target.elevation * pi / 180);
        point.intensity = target.level;

        p2gScan.points.push_back(point);

      }

    }


    /** Reads datastream into a private class variable to avoid creating
     *  a buffer object on each poll. Assumes serial connection is alive
     *  when function is called. The high frequency at which we poll
     *  necessitates that we dispense with edge-case checks.
     */
    bool Sensor::poll(void) {

      int res = -1;

      if (type == "target") {
        targetScan.header.stamp = ros::Time::now();
        targetScan.header.frame_id = _frame_id;
        targetScan.targets.clear();

        p2gScan.header.stamp = ros::Time::now();
        p2gScan.header.frame_id = _frame_id;
        p2gScan.points.clear();

        res = ep_targetdetect_get_targets(_protocolHandle, _endpoint);

        if (res >= 0) {

          _pub.publish(targetScan);
          _point_pub.publish(p2gScan);
        } else return false;

      } else { // case "base":
        baseScan.header.stamp = ros::Time::now();
        baseScan.header.frame_id = _frame_id;
        baseScan.chirps.clear();

        res = ep_radar_base_get_frame_data(_protocolHandle, _endpoint, 1);

        if (res < 0)
          return false;
      }

      return true;
    }

/** Deletes underlying serial and config server objects
 *  managed by class pointers.
 */
    void Sensor::shutdown() {
      protocol_disconnect(_radar_handle);
      _srv.reset();
    }

/** Only parameters within the root group of cfg ParameterGenerator
 *  broadcast their default values on initialization, so this method
 *  only transmits them to the sensor.
 *
 *  Parameters in any sub-groups broadcast their value as 0 on startup.
 *  It is also possible that sub-group params do not broadcast their
 *  value on startup, so polling the cfg server returns 0.
 *  This is likely a bug with the ROS dynamic reconfigure library.
 */
    void Sensor::_init(void) {
      bool success = true;

      ROS_INFO("Initializing.");

      for (int i = 0; i < 43; i += 10) {
        ROS_INFO_STREAM("Update parameter level " << i << std::endl);
        this->_reconfigure(_cfg, i);
      }

      if (success) ROS_INFO("Sensor settings initialized");
      else ROS_WARN("One or more settings failed to initialize");
    }

    /** Determines which setting has changed and transmits the associated
     *  (well-formed) settings command to the serial stream. A unique level
     *  is assigned to each settings parameter in the cfg file. Current
     *  implementation defines 12 sensor performance parameters, indexed in
     *  cfg from 0 to 11. Config server triggers this method upon initialization
     *  with a special level of -1.
     */
    void Sensor::_reconfigure(p2gConfig &cfg, uint32_t level) {

      if ((int) level > 44 || (int) level < -1) {
        ROS_INFO("Update skipped: Parameter not recognized");
        return;
      }

      _cfg = cfg;

      if (level == -1) {
        this->_init();
      }


      if (level >= 40) {
        Frame_Format_t firmware_cfg;

        firmware_cfg.num_chirps_per_frame = 4; // cfg.base_num_chirps_per_frame;
        firmware_cfg.num_samples_per_chirp = 256; // cfg.base_num_samples_per_chirp;
        firmware_cfg.rx_mask = cfg.base_rx_mask;

        auto status_code = ep_radar_base_set_frame_format(_protocolHandle, _endpoint, &firmware_cfg);
        ROS_INFO_STREAM(protocol_get_status_code_description(_protocolHandle, status_code));

      } else if (level >=
                 20) { // --------------------------------------------------------------------------------------
        DSP update
        if (ep_targetdetect_is_compatible_endpoint(_protocolHandle, _endpoint) == 0) {

          DSP_Settings_t firmware_cfg;

          firmware_cfg.range_mvg_avg_length = cfg.dsp_range_mvg_avg_length;
          firmware_cfg.range_thresh_type = cfg.dsp_range_thresh_type;
          firmware_cfg.min_range_cm = cfg.dsp_min_range_cm;
          firmware_cfg.max_range_cm = cfg.dsp_max_range_cm;
          firmware_cfg.min_speed_kmh = cfg.dsp_min_speed_kmh;
          firmware_cfg.max_speed_kmh = cfg.dsp_max_speed_kmh;
          firmware_cfg.range_threshold = cfg.dsp_range_threshold;
          firmware_cfg.speed_threshold = cfg.dsp_speed_threshold;
          firmware_cfg.adaptive_offset = cfg.dsp_adaptive_offset;
          firmware_cfg.enable_tracking = cfg.dsp_enable_tracking;
          firmware_cfg.num_of_tracks = cfg.dsp_num_of_tracks;
          firmware_cfg.median_filter_length = cfg.dsp_median_filter_length;
          firmware_cfg.enable_mti_filter = cfg.dsp_enable_mti_filter;
          firmware_cfg.mti_filter_length = cfg.dsp_mti_filter_length;

          if (ep_targetdetect_set_dsp_settings(_protocolHandle, _endpoint, &firmware_cfg) == -1) {
            ROS_WARN("Parameter setting command returned failure!");
          };
        } else {
          ROS_WARN("Connected to wrong target. Settings of type dsp cannot be applied. Aborting ...");
        }
      } else if (level >=
                 10) { // -------------------------------------------------------------------------------------
        PGA update
        if (ep_radar_p2g_is_compatible_endpoint(_protocolHandle, _endpoint) == 0) {
          if (ep_radar_p2g_set_pga_level(_protocolHandle, _endpoint, cfg.pga_level) == -1) {
            ROS_WARN("Parameter setting command returned failure!");
          };
        } else {
          ROS_WARN("Connected to wrong target. Settings of type pga cannot be applied. Aborting ...");
        }
      } else if (level >=
                 0) { // --------------------------------------------------------------------------------------------
        // ADC update
        //if (ep_radar_adcxmc_is_compatible_endpoint(_protocolHandle, _endpoint) == 0) {

        Adc_Xmc_Configuration_t firmware_cfg;
        firmware_cfg.samplerate_Hz = cfg.adc_samplerate_hz;
        firmware_cfg.resolution = cfg.adc_resolution;
        firmware_cfg.use_post_calibration = cfg.adc_use_post_calibration;

        if (ep_radar_adcxmc_set_adc_configuration(_protocolHandle, _endpoint, &firmware_cfg) == -1) {
          ROS_WARN("Parameter setting command returned failure!");
        };
      } else {
        ROS_WARN("Connected to wrong target. Settings of type adc cannot be applied. Aborting ...");
      }
    }

}


} // namespace p2g_ros_driver
