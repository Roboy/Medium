#!/usr/bin/env python

import numpy as np
import torch
import roslib
import rospy
import sys
from toposens_msgs.msg import TsScan

import tf2_ros as tf
import tf_conversions
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String, Int16MultiArray
from sensor_msgs.msg import Image, CompressedImage
from medium_localizer.srv import *
from cv_bridge import CvBridge

import matplotlib.pyplot as plt

import json
from plotter import *
from plotter_wlb import *

# OpenCV
import cv2

class KeypointsVisualizerNode():

    def __init__(self):
        self.pub = rospy.Publisher('human_poses', String, queue_size=2)
        self.persons_close = False

        # Publishers
        self.plt = Plotter(image_h=480, image_w=640, scale_factor=10)
        self.plt_wlb = WlbPlotter()
        self.bridge = CvBridge()
        self.loc_viz_publisher = rospy.Publisher("/medium/viz/stickman", Image, queue_size=2)
        self.wlb_viz_publisher = rospy.Publisher("/medium/viz/raw_wlb", Image, queue_size=2)
        self.wlb_viz_slice_publisher = rospy.Publisher("/medium/viz/wlb_slice", Image, queue_size=2)

        # Subscribers
        # self._sub_scans = rospy.Subscriber('/ts_scans', TsScan, self._add_scan, queue_size=10)
        self.wlb_subscriber = rospy.Subscriber("/wlb/raw_image", Int16MultiArray, self.callback_wlb_raw_images_mc, queue_size=10)
        self.wlb_slice_subscriber = rospy.Subscriber("/wlb/raw_image/slice", Int16MultiArray, self.callback_wlb_raw_images_slices, queue_size=10)
        # rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback, queue_size=10)

        # tf Broadcasters
        self._tf_broadcaster_dynamic = tf.TransformBroadcaster()
        self._tf_broadcaster_static = tf.StaticTransformBroadcaster()

        # TODO: Can we get the frame_id names from the parameter server?
        # self._t_world2ts30 = self._initialize_stamped_transform(parent_frame='camera', child_frame='toposens')

        # Get tf to listen
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        # Cache received messages
        self.captured_scans = np.array([])
        self.captured_wlb = np.array([])

    def request_radar_keypoints(self):
        """
        # Service call to 'localize keypoints' and publishes a visualization of the keypoints
        :return:
        """
        rospy.wait_for_service('localize_keypoints')
        try:
            service = rospy.ServiceProxy('localize_keypoints', LocalizeKeypoints)
            req = service(0)

            keypoints = json.loads(req.keypoints)

            # if not keypoints:

            self.plt.set_keypoints(keypoints)
            image = None
            # image = self.plt.draw_keypoints(image)
            image = self.plt.plot_stick_man(image)[:, :, :3]

            if 'Nose' in keypoints.keys():
                image = self.plt.plot_robot_head(image)

            if image.max() > 0:
                image *= (255/image.max())

            msg = self.bridge.cv2_to_imgmsg(image.astype(np.uint8))
            self.loc_viz_publisher.publish( msg )

        except rospy.ServiceException as e:
            print("Service call failed: %s", e)

    def callback(self, ros_data):
        """
        Overlay pose at torso location onto camera image.
        :param ros_data:
        :return:
        """
        global x_history, y_history, pose_history

        # Conversion to CV2
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rospy.wait_for_service('localize_keypoints')
        try:
            service = rospy.ServiceProxy('localize_keypoints', LocalizeKeypoints)
            req = service(0)

            keypoints = json.loads(req.keypoints)

            self.plt.set_keypoints(keypoints)
            # image_np = self.plt.draw_keypoints(image_np)
            image = self.plt.plot_stick_man(image_np)[:, :, :3]
            if 'Nose' in keypoints.keys():
                image = self.plt.plot_robot_head(image)

            plt.imshow(image)
            # cv2.imwrite("frame%d.jpg" % count, image)
            # cv2.waitKey(2)
            plt.show()

        except rospy.ServiceException as e:
            print("Service call failed: %s", e)



    def callback_wlb_raw_images_mc(self, msg):
        """
        Callback for walabaot raw images, publishes a visualization of the raw images on /wlb/raw_images
        :param msg:
        :return:
        """

        raw_image = np.array(msg.data).reshape((7, 25, 116))

        image_np = self.plt_wlb.plot_marching_cubes([raw_image])
        # image_np = self.plt_wlb.plot_raw_measurements(raw_image)

        msg = self.bridge.cv2_to_imgmsg(image_np.astype(np.uint8))
        # msg.header.frame_id = 'walabot'

        self.wlb_viz_publisher.publish(msg)

    def callback_wlb_raw_images_slices(self, msg):
        """
        Callback for walabaot raw images, publishes a visualization of the raw images on /wlb/raw_images
        :param msg:
        :return:
        """

        image = np.array(msg.data).reshape((116, 25))

        # msg.header.frame_id = 'walabot'

        image_np = self.plt_wlb.plot_raw_image_slice(image)

        msg = self.bridge.cv2_to_imgmsg(image_np.astype(np.uint8))

        self.wlb_viz_slice_publisher.publish(msg)

    def callback_wlb_raw_images(self, msg):
        """
        Callback for walabaot raw images, publishes a visualization of the raw images on /wlb/raw_images
        :param msg:
        :return:
        """

        raw_image = np.array(msg.data).reshape((7, 25, 116))

        # image_np = self.plt_wlb.plot_marching_cubes([raw_image])
        image_np = self.plt_wlb.plot_raw_measurements(raw_image)

        msg = self.bridge.cv2_to_imgmsg(image_np.astype(np.uint8))
        # msg.header.frame_id = 'walabot'

        self.wlb_viz_publisher.publish(msg)

    def _add_scan(self, msg):
        """
        Callback for toposens sensor subscriber. Only adds scans when flag is set.
        :param msg: ROS toposens_msgs package's TsScan message type
        """
        # Transform points in ts_scans message into world frame and convert to array
        # Timestamp, X, Y, Z, V
        ts_points = self._extract_points(msg)

        if np.array(ts_points).shape[0] == 0:
            return

        if self.captured_scans is None or self.captured_scans.shape == ():
            self.captured_scans = ts_points
        else:
            for ts_point in ts_points:
                if np.linalg.norm(ts_point[1:4]) > 0.5: # discard ego reflecations
                    self.captured_scans = np.append(self.captured_scans, ts_point.reshape((1, 8)), axis=0)

            self.captured_scans = self.captured_scans[-20:]

            if np.linalg.norm(self.captured_scans[-5:, 1:4]) < 4:
                rospy.logwarn("Too close for my taste in radar!")


    def _initialize_stamped_transform(self, parent_frame, child_frame):
        """
        Initializes a stamped transform message with frame_ids.
        :param parent_frame: frame_id of the parent in the tf tree as a string
        :param child_frame: frame_id of the child in the tf tree as a string
        :return: initialized tf stamped transform message
        """
        t = TransformStamped()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        return t

    def _extract_points(self, msg):
        """
        Extracts the scan points from the captured message both in sensor COS and transforms
        them into world coordinates given the current tf transformations.
        :param msg: TsScan message
        :return: numpy array of points [timestamp, x_s, y_s, z_s, x_w, y_w, z_w, v],
                    where index s indicates sensor COS and index w indicates world COS
        """
        points = np.zeros((len(msg.points), 8))
        # transform_msg = self.tf_buffer.lookup_transform("camera", msg.header.frame_id, rospy.Time())

        for i, point in enumerate(msg.points):
            # point_transformed = _apply_transformation([point.location.x, point.location.y, point.location.z],
            #                                           transform_msg.transform)
            # Timestamp, X_s, Y_s, Z_s, X_w, Y_w, Z_w, V
            points[i, :] = [msg.header.stamp.to_nsec(),
                            point.location.x, point.location.y, point.location.z,
                            # point_transformed[0], point_transformed[1], point_transformed[2],
                            point.location.x, point.location.y, point.location.z-0.15,
                            point.intensity]

        return points


def _apply_transformation(point, transform):
    """
    Converts the tf rotation quaternion to a rotation matrix and applies it with a
    subsequent translation to a point.
    :param point: point coordinates as a list of [x,y,z]
    :param transform: tf transformation
    :return: transformed point [x',y',z']
    """
    point.append(1)  # Add 4th coordinate for homogeneous coordinates

    quat = [transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w]

    matrix = quaternion_matrix(quat)

    point_rotated = np.dot(matrix, point)
    point_translated = [point_rotated[0] + transform.translation.x,
                        point_rotated[1] + transform.translation.y,
                        point_rotated[2] + transform.translation.z]

    return point_translated


if __name__ == "__main__":
    rospy.init_node('medium_visualizer_node', anonymous=True)

    visualizer = KeypointsVisualizerNode()

    rate = rospy.Rate(5)  # 1hz


    while not rospy.is_shutdown():
        visualizer.request_radar_keypoints()
        rate.sleep()
