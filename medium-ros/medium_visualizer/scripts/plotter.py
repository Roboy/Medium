import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

keypoint_pairs = [
    [('Neck', 'RShoulder'), None, [255, 102, 102]],
    [('Neck', 'LShoulder'), None, [255, 102, 102]],
    [('RShoulder', 'RElbow'), None, [255, 128, 0]],
    [('RElbow', 'RWrist'), None, [255, 255, 50]],
    [('LShoulder', 'LElbow'), None, [255, 128, 0]],
    [('LElbow', 'LWrist'), None, [255, 255, 50]],
    [('Neck', 'RHip'), None, [0, 150, 0]],
    [('RHip', 'RKnee'), None, [0, 204, 204]],
    [('RKnee', 'RAnkle'), None, [51, 51, 255]],
    [('Neck', 'LHip'), None, [0, 150, 0]],
    [('LHip', 'LKnee'), None, [0, 204, 204]],
    [('LKnee', 'LAnkle'), None, [51, 51, 255]],
    [('Neck', 'Nose'), None, [153, 0, 0]],
    [('Nose', 'REye'), None, [153, 0, 153]],
    [('REye', 'REar'), None, [153, 0, 153]],
    [('Nose', 'LEye'), None, [153, 0, 153]],
    [('LEye', 'LEar'), None, [153, 0, 153]],
    [('RShoulder', 'REar'), None, [153, 0, 153]],
    [('LShoulder', 'LEar'), None, [153, 0, 153]],
    [('LHip', 'LAnkle'), 'LKnee', [255, 255, 255]],
    [('RHip', 'RAnkle'), 'RKnee', [255, 255, 255]],
    [('RShoulder', 'RAnkle'), 'RElbow', [255, 255, 255]],
    [('LShoulder', 'LAnkle'), 'LElbow', [255, 255, 255]],
    [('RHip', 'LHip'), 'Neck', [255, 255, 255]],
    [('RHip', 'RShoulder'), 'Neck', [255, 255, 255]],
    [('LHip', 'LShoulder'), 'Neck', [255, 255, 255]],
]

robot_pairs = {
    'arm_right': [('RShoulder', 'RWrist'), None, [255, 255, 50]],
    'arm_left': [('LShoulder', 'LWrist'), None, [255, 128, 0]],
    'leg_right': [('RHip', 'RAnkle'), None, [0, 204, 204]],
    'leg_left': [('LHip', 'LAnkle'), None, [0, 204, 204]],
}

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


class Plotter:
    def __init__(self, image_h=64, image_w=64, scale_factor=1):

        self.scale_factor = scale_factor
        self.image_h = image_h
        self.image_w = image_w
        self.centers = {}
        self.body_pred = {}

        full_path = os.path.realpath(__file__)
        self.dirpath = os.path.dirname(full_path)
        # TODO: find pictures for all robot parts
        self.robot_parts = {'arm_right': {'img': cv2.imread(os.path.join(self.dirpath, "/robot_arm.png"), -1),
                                          'centers': np.array([600, 0])
                                          },  # vector in length of arm
                            'arm_left': {'img': cv2.imread(os.path.join(self.dirpath, "/robot_arm.png"), -1),
                                         'centers': np.array([-600, 0])
                                         },  # vector in length of arm
                            'leg_right': {'img': cv2.imread(os.path.join(self.dirpath, "/robot_arm.png"), -1),
                                          'centers': np.array([600, 0])
                                          },  # vector in length of arm
                            'leg_left': {'img': cv2.imread(os.path.join(self.dirpath, "/robot_arm.png"), -1),
                                         'centers': np.array([-600, 0])
                                         },  # vector in length of arm
                            'face': {'img': cv2.imread(os.path.join(self.dirpath + "/einstein.png"), -1),
                                     'centers': np.array([256, 256])
                                     },
                            }

    def set_keypoints(self, keypoints):
        """
        Set centers of keypoints in image from keypoints dict.
        :param keypoints:
        :return:
        """

        centers = {}
        # draw point
        for key in keypoint_columns:
            if key not in keypoints.keys():
                continue

            keypoint = keypoints[key]
            center = (int(keypoint[0] * self.image_w), int(keypoint[1] * self.image_h))
            self.centers[key] = center

        return centers

    def draw_keypoints(self, img=None):
        """
        Draw circles around detected keypoints into image.
        :param y_pred:
        :param img:
        :return:
        """
        if img is None:
            img = np.zeros((self.image_h, self.image_w, 3))

        if not self.centers:
            cv2.putText(img,
                        'No reliable keypoints detected.',
                        (20, self.image_h-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

        # draw points
        for key in keypoint_columns:
            if key not in self.centers.keys():
                continue

            cv2.circle(img, (self.centers[key][0], self.centers[key][1]), radius=self.scale_factor, thickness=-1,
                       color=[0, 255, 0])

        return img

    def plot_stick_man(self, img=None):
        """
        Connect keypoints by colored lines to build a human body out of lines.
        :param y_pred:
        :return:
        """
        if img is None:
            img = np.zeros((self.image_h, self.image_w, 3)).astype(np.uint8)

        if not self.centers:
            cv2.putText(img,
                        'No reliable keypoints detected.',
                        (20, self.image_h-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

        # draw line
        for pair_order, pair in enumerate(keypoint_pairs):
            if pair[0][0] not in self.centers.keys() \
                    or pair[0][1] not in self.centers.keys() \
                    or pair[1] in self.centers.keys():
                continue

            radius = np.linalg.norm(np.array(self.centers[pair[0][1]]) - np.array(self.centers[pair[0][0]]))

            if radius < self.image_h / 6:
                cv2.line(img, self.centers[pair[0][0]], self.centers[pair[0][1]], pair[2], self.scale_factor)

        return img

    def plot_robot_head(self, img):
        """
        Overlay roboy Einstein head at position of nose keypoint.
        :param y_pred:
        :param img:
        :return: img
        """
        img = img.astype(np.uint8)

        dist_LShoulder = np.linalg.norm(np.array(self.centers.get('LShoulder', [0,0]))
                                                 - np.array(self.centers.get('Nose', [1000, 0])))
        dist_RShoulder = np.linalg.norm(np.array(self.centers.get('RShoulder', [0,0]))
                                                 - np.array(self.centers.get('Nose', [1000, 0])))

        # print("dist lshoulder {}, dist rshoulder {}".format(dist_LShoulder, dist_RShoulder))

        if dist_LShoulder > 0.1 * self.image_h or dist_RShoulder > 0.1 * self.image_h:
            return img

        robot_part = self.robot_parts['face']
        s_img = robot_part['img'][:, :, :3]
        s_vec = robot_part['centers']

        # Scale robot part
        shrink_factor = 0.2
        s_img = cv2.resize(s_img,
                           None,
                           fx=shrink_factor,
                           fy=shrink_factor,
                           interpolation=cv2.INTER_CUBIC)

        x_nose = self.centers['Nose'][0] + 5
        y_nose = self.centers['Nose'][1] + 5

        y1, y2 = int(y_nose - s_img.shape[0] / 2), int(y_nose + s_img.shape[0] / 2)
        x1, x2 = int(x_nose - s_img.shape[1] / 2), int(x_nose + s_img.shape[1] / 2)

        roi = img[y1:y2, x1:x2]
        try:
            # Now create a mask of logo and create its inverse mask also
            s_img2gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(s_img2gray, 254, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of logo in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask)

            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(s_img, s_img, mask=mask_inv)

            # Put logo in ROI and modify the main image
            dst = cv2.add(img_bg, img2_fg)
            img[y1:y2, x1:x2] = dst
        except Exception as e:
            print("Error in plot_robot_head: " + e.message)

        return img

    def plot_robot_man(self, img=None):
        """
        Basically the same as stick man, but overlays pictures of robotic arms and legs connecting the keypoints.
        :param y_pred:
        :param img:
        :return:
        """

        if img is None:
            img = np.zeros((self.image_w, self.image_h, 3))

        if not self.centers:
            cv2.putText(img,
                        'No reliable keypoints detected.',
                        (20, self.image_h-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

        # draw line
        for key, pair in robot_pairs.items():
            if pair[0][0] not in self.centers.keys() or pair[0][1] not in self.centers.keys() or pair[1] in self.centers.keys():
                continue

            try:
                # Get body part
                robot_part = self.robot_parts[key]  # [:,:,:3]
                s_img = robot_part['img']
                s_vec = robot_part['centers']

                # inner product between keypoints vector and img
                keypoint_vector = np.array([self.centers[pair[0][0]][1] - self.centers[pair[0][1]][1],
                                            self.centers[pair[0][0]][0] - self.centers[pair[0][1]][0]])
                inner_product = np.dot(keypoint_vector / np.linalg.norm(keypoint_vector), s_vec / np.linalg.norm(s_vec))
                scale_factor = np.linalg.norm(keypoint_vector) / np.linalg.norm(s_vec)

                # Rotate robot part
                if scale_factor == 0:
                    continue

                (rows, cols, _) = s_img.shape
                R = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                            angle=np.rad2deg(np.arccos(inner_product)),
                                            scale=1
                                            )
                s_img = cv2.warpAffine(s_img, R, (cols, rows))

                # Scale robot part
                s_img = cv2.resize(s_img,
                                   None,
                                   fx=scale_factor,
                                   fy=scale_factor,
                                   interpolation=cv2.INTER_CUBIC)

                # Place limb at correct position in image
                img_x1 = np.minimum(self.centers[pair[0][1]][0] + keypoint_vector[0],
                                    self.centers[pair[0][1]][0] - keypoint_vector[0])
                img_x2 = np.maximum(self.centers[pair[0][1]][0] + keypoint_vector[0],
                                    self.centers[pair[0][1]][0] - keypoint_vector[0])
                img_y1 = np.minimum(self.centers[pair[0][1]][1] + keypoint_vector[1],
                                    self.centers[pair[0][1]][1] - keypoint_vector[1])
                img_y2 = np.maximum(self.centers[pair[0][1]][1] + keypoint_vector[1],
                                    self.centers[pair[0][1]][1] - keypoint_vector[1])

                x_avg = np.mean([img_x1, img_x2])
                y_avg = np.mean([img_y1, img_y2])

                y1, y2 = int(y_avg - s_img.shape[0] / 2), int(y_avg + s_img.shape[0] / 2)
                x1, x2 = int(x_avg - s_img.shape[1] / 2), int(x_avg + s_img.shape[1] / 2)

                img[y1:y2, x1:x2, :] = s_img[:, :, :3]

            except Exception as e:
                print("Error in plot_robot_man: " + e.message)

        return img
s