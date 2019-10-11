#include "p2g_ros_markers/plot.h"
#include <p2g_ros_driver/sensor.h>


namespace p2g_ros_markers
{
/** A dynamic reconfigure server is set up to change marker scale
 *  and lifetime during runtime.
 */
Plot::Plot(ros::NodeHandle nh, ros::NodeHandle private_nh)
{
  private_nh.param<std::string>("frame_id", _frame_id, "position2go");		//Frame for tf
  _rviz.reset(new rviz_visual_tools::RvizVisualTools(_frame_id, "/" + kMarkersTopic));

  pub = nh.advertise<visualization_msgs::MarkerArray> (kMarkersTopic, 5);

	// Subscribe to topic with TS scans
	_scans_sub = nh.subscribe(p2g_ros_driver::kTargetScansTopic, 100, &Plot::_sanitize, this);
}

/** All incoming scans are stored in a double-ended queue, which
 *  allows for easy iteration, sequential arrangement of data and
 *  efficient chronological cleaning in subsequent runs. On each
 *  trigger, all scans that have expired their lifetime are deleted.
 *  Rviz plotting is thereafter refreshed for this sanitized data.
 */
void Plot::_sanitize(const p2g_ros_msgs::TargetScans::ConstPtr& msg)
{

  _scans.push_back(*msg);

  do {
    ros::Time oldest = _scans.front().header.stamp;
    // break loop if oldest timestamp is within lifetime
		if (ros::Time::now() - oldest < ros::Duration(1)) break;
    _scans.pop_front();
  } while(!_scans.empty());

  Plot::_plot();
}

/** Rviz provides no straightforward way of deleting individual
 *  markers once they have been plotted. Refreshing of display is
 *  achieved by clearing all markers on each run and re-plotting
 *  all data contained in the most recent sanitized copy of incoming
 *  scans.
 *
 *  Sensing range is also updated to keep track of the furthest point
 *  detected by the sensor. This is used for color-coding the markers.
 *
 *  RVT (Rviz Visual Tools) wrappers are used for efficient Rviz
 *  plotting. Since batch publishing is enabled, RVT collects all
 *  markers to be visualized in a given update and publishes them
 *  to Rviz in one go when trigger() is called.
 */
void Plot::_plot(void) {

  visualization_msgs::MarkerArray markerArray;

  for (auto it1 = _scans.begin(); it1 != _scans.end(); ++it1) {
		std::vector<p2g_ros_msgs::Target> targets = it1->targets;

    for (auto it2 = targets.begin(); it2 != targets.end(); ++it2) {
			p2g_ros_msgs::Target target = *it2;

      visualization_msgs::Marker marker;

      marker.header.frame_id = "position2go";
      marker.header.stamp = ros::Time(0);
      marker.ns = "position2go";
      marker.id = target.target_id;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = target.radius/100 * std::cos(target.elevation *pi/180) * std::cos(target.azimuth *pi/180);
      marker.pose.position.y = target.radius/100 * std::cos(target.elevation *pi/180) * std::sin(target.azimuth *pi/180);
      marker.pose.position.z = target.radius/100 * std::sin(target.elevation *pi/180);
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.5 + target.level/5000;
      marker.scale.y = 0.5 + target.level/5000;
      marker.scale.z = 0.5 + target.level/5000;

      std_msgs::ColorRGBA color = _rviz->getColorScale(marker.pose.position.x/10);

      marker.color.a = color.a;
      marker.color.r = color.r;
      marker.color.g = color.g;
      marker.color.b = color.b;
      marker.lifetime = ros::Duration(1);

      markerArray.markers.push_back(marker);
    }
  }

      pub.publish(markerArray);
}


} // namespace p2g_ros_markers
