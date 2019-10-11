#!/usr/bin/env python

import glob
import json
import os

# OpenCV
import numpy as np

# ROS libraries
import rospy
from Net_model import *
from medium_localizer.srv import LocalizeKeypoints
from std_msgs.msg import Int16MultiArray

# PyTorch
import torch
from torchvision import transforms

cuda = torch.device('cuda')  # Default CUDA device
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

keypoint_columns = [
    'LAnkle',
    'LElbow',
    'LHip',
    'LKnee',
    'LShoulder',
    'LWrist',
    'Nose',
    'RAnkle',
    'RElbow',
    'RHip',
    'RKnee',
    'RShoulder',
    'RWrist'
]


class KeypointsLocalizer:

    def __init__(self, num_boards=4, num_samples=64):
        my_path = os.path.abspath(os.path.dirname(__file__))

        self.Scan = -np.ones((2, 4 * 2, num_samples))
        self.raw_images_queue = []

        # Subscribers
        # rospy.Subscriber('/p2g_base_scans', BaseScans, self.callback_p2g_scan, queue_size=1, buff_size=2 ** 24)
        self.wlb_subscriber = rospy.Subscriber("/wlb/raw_image", Int16MultiArray, self.callback_wlb_raw_images,
                                               queue_size=10)

        # Services
        self.service = rospy.Service('localize_keypoints', LocalizeKeypoints, self.infer_radar_keypoints)

        full_path = os.path.realpath(__file__)
        self.dirpath = os.path.dirname(full_path)

        model_path = glob.glob(self.dirpath + "/" + "*.pt")[0]
        rospy.loginfo("Loaded model parameters from: " + model_path)

        self.model = RF_Pose3D_130100_Dropout()
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def infer_radar_keypoints(self, req):
        """
        Callback that serves 'localize keypoints' service. Infers keypoints from accumulated walabot raw_images queue.
        :param req:
        :return:
        """
        input = np.array(self.raw_images_queue)

        if len(self.raw_images_queue) < 20:
            return json.dumps({})

        if model_type == 'Double':
            x1 = 
            
            trans = transforms.ToTensor()
            input_tensor = torch.from_numpy(input)
            input_tensor = input_tensor.float()
            input_tensor = input_tensor.to(device)
            input_tensor = input_tensor.unsqueeze(dim=0)
            predictions = self.model(input_tensor)
        else:
            trans = transforms.ToTensor()
            input_tensor = torch.from_numpy(input)
            input_tensor = input_tensor.float()
            input_tensor = input_tensor.to(device)
            input_tensor = input_tensor.unsqueeze(dim=0)
            predictions = self.model(input_tensor)
        
        keypoints = extract_keypoints_max(predictions.cpu())
        # with torch.no_grad():
        #   keypoints = extract_keypoints_avg(predictions.cpu().detach().numpy())

        return json.dumps(keypoints)

    def callback_wlb_raw_images(self, msg):
        """
        Callback for wlb raw_image topic. Accumulates received images in a queue of fixed length.
        :param msg:
        :return:
        """

        raw_image = np.array(msg.data).reshape((7, 25, 116))

        self.raw_images_queue.append(raw_image.tolist())
        self.raw_images_queue = self.raw_images_queue[-20:]


def extract_keypoints_max(predictions):
    """
    Extract keypoitns of heatmaps based on maximum confidence.
    :param predictions:
    :return:
    """
    height = predictions.shape[3]
    width = predictions.shape[2]

    keypoints = {}
    fail_counter = 0
    for j, keypoint in enumerate(keypoint_columns):
        (x, y) = list(np.unravel_index(predictions[0, j, :, :].argmax(), predictions[0, j, :, :].shape))
        if (x == 0 and y == 0):  # or (predictions[0, j, x, y].detach().numpy() < 0.18):
            keypoints[keypoint] = [-1, -1]  # not found
            fail_counter = fail_counter + 1

        else:
            keypoints[keypoint] = [(float(x) / width), float(y) / height]

        # print("Confidence: {}".format(predictions[0, j, x, y].detach().numpy()))
        if predictions[0, j, x, y].detach().numpy() <= 0.25:
            fail_counter = fail_counter + 1

    if fail_counter <= 6:
        return keypoints
    else:
        return {}


def extract_keypoints_avg(predictions, threshold=0.1):
    """
    Computes weighted average center.
    :param predictions:
    :return:
    """
    threshold = np.minimum(predictions.max(), threshold)

    keypoints = {}
    
    for j, keypoint in enumerate(keypoint_columns):
        arr = predictions[0, :, :, j]

        mask_out_indices = arr < threshold
        arr[mask_out_indices] = 0

        width = arr.shape[0]
        height = arr.shape[1]

        x_range = range(0, width)
        y_range = range(0, height)

        (X, Y) = np.meshgrid(x_range, y_range)

        arr_sum = arr.sum()

        if arr_sum == 0:
            continue  # no keypoints found

        x = (np.multiply(X, arr)).sum() / arr_sum
        y = (np.multiply(Y, arr)).sum() / arr_sum

        keypoints[keypoint] = [(float(x) / width), float(y) / height]
    return keypoints


if __name__ == '__main__':
    rospy.init_node('keypoints_localizer_node', anonymous=True, log_level=rospy.INFO)

    localizer = KeypointsLocalizer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down keypoint localizer module")
