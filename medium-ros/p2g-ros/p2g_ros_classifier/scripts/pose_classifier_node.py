#!/usr/bin/env python
import time
import os
import sys
import ast

from collections import Counter

# Ros libraries
import roslib
import rospy
import rospy
import rospkg

# Ros messages
from std_msgs.msg import String
from p2g_ros_msgs.msg import P2GScan, BaseScans, Chirp, Antenna, Sample
from sensor_msgs.msg import CompressedImage

os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import csv
import keras
from keras.layers import Dense, Activation

# OpenCV
import cv2

VERBOSE = True
x_history = []
y_history = []
pose_history = []


class TorsoLocalizer:

    def __init__(self):

        my_path = os.path.abspath(os.path.dirname(__file__))

        # Load network for torso localization as image pixel
        self.torso_localizer = keras.Sequential([
            Dense(256, input_shape=[64 * 16 * 2]),
            Activation("relu"),
            Dense(128, ),
            Activation("relu"),
            Dense(128, ),
            Activation("relu"),
            Dense((32 * 32)),
        ])
        self.torso_localizer.load_weights(os.path.join(my_path, "latest_torso_localizer_weights.h5"))
        self.torso_location = None

        # Load network trained for pose estimation
        self.pose_estimator = keras.Sequential([
            Dense(256, input_shape=(64 * 16 * 2,)),
            Activation("relu"),
            Dense(128, ),
            Activation("relu"),
            Dense(128, ),
            Activation("relu"),
            Dense(5),
            Activation("softmax")
        ])
        self.pose_estimator.load_weights(os.path.join(my_path, "latest_poseclassifier_weights.h5"))
        self.pose = None


        # subscribe to radar samples and camera messages
        rospy.Subscriber('p2g_base_scans', BaseScans, self.callback_p2g_scan, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback, queue_size=1)

        # Load images for pose visualization
        self.pose_images = {0: cv2.imread(os.path.join(my_path, "roboy_bad_pose.png"), -1),
                            1: cv2.imread(os.path.join(my_path, "roboy_arms_down.png"), -1),
                            2: cv2.imread(os.path.join(my_path, "arms_180.png"), -1),
                            3: cv2.imread(os.path.join(my_path, "arms_90.png"), -1),
                            4: cv2.imread(os.path.join(my_path, "egypt.jpg"), -1)}

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

        if self.torso_location:
            x_history.append(self.torso_location[0] * image_np.shape[1] / 32)
            y_history.append(self.torso_location[1] * image_np.shape[0] / 32)
            pose_history.append(self.pose)

            print(self.pose)

        # Implement moving average over all predictions
        average_length = 20

        if len(x_history) > average_length:
            x_history = x_history[-average_length:]
            y_history = y_history[-average_length:]
            pose_history = pose_history[-average_length:]

        if x_history:
            x_avg = sum(x_history) / float(len(x_history))
            y_avg = sum(y_history) / float(len(y_history))

            c = Counter(pose_history)
            pose_maj, _ = c.most_common()[0]

        if self.torso_location:
            s_img = self.pose_images[pose_maj]  # [:,:,:3]

            y1, y2 = int(y_avg - s_img.shape[0] / 2), int(y_avg + s_img.shape[0] / 2)
            x1, x2 = int(x_avg - s_img.shape[1] / 2), int(x_avg + s_img.shape[1] / 2)

            image_np[y1:y2, x1:x2, :] = s_img[:, :, :3]

        cv2.imshow('cv_img', image_np)
        cv2.waitKey(2)

    def callback_p2g_scan(self, p2g_scan):

        # Extract radar samples and transform them into radar matrices
        rd_maps = {}

        frame_antenna0 = []
        frame_antenna1 = []

        for chirp_msg in p2g_scan.chirps:
            chirp = []
            for antenna_msg in chirp_msg.antennas:
                real = []
                imag = []
                for sample in antenna_msg.samples:
                    real.append(sample.real)
                    imag.append(sample.imag)

                antenna = {'real': real,
                           'imag': imag}

                chirp.append(antenna)

            frame_antenna0.append(np.array(chirp[0]['real']) + 1j * np.array(chirp[0]['imag']))
            frame_antenna1.append(np.array(chirp[1]['real']) + 1j * np.array(chirp[1]['imag']))

        rd_maps['antenna0'] = self.compute_rd_maps(frame_antenna0)
        rd_maps['antenna1'] = self.compute_rd_maps(frame_antenna1)

        input_flatten = np.array([rd_maps['antenna0'], rd_maps['antenna1']]).reshape(1, 2048)

        # Input radar samples into torso localization network
        prediction = self.torso_localizer.predict(input_flatten)
        prediction = prediction.reshape(-1, 32, 32)[0]

        self.torso_location = np.unravel_index(prediction.argmax(), prediction.shape)

        # Input radar samples to pose estimation network
        pose_prediction = self.pose_estimator.predict_classes(input_flatten)
        self.pose = pose_prediction[0]



    def compute_rd_maps(self, frame):

        range_matrix = []
        for chirp in frame:
            range_matrix.append(np.fft.fft(chirp - np.mean(chirp)))

        rd_map = []
        for column in np.array(range_matrix).transpose():
            rd_map.append(np.fft.fftshift(np.fft.fft(column)))

        return np.array(rd_map)  # .transpose()


if __name__ == '__main__':
    rospy.init_node('p2g_pose_estimator_node', anonymous=True, log_level=rospy.INFO)

    tl = TorsoLocalizer()
    # pub_pose = rospy.Publisher('p2g_pose', Pose, queue_size=1)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    cv2.destroyAllWindows()
