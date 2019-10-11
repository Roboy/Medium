import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class Plotter:
    def __init__(self, image_h=64, image_w=64, scale_factor=1):
        self.keypoint_pairs = [
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

        self.robot_pairs = {
            'arm_right':[('RShoulder', 'RWrist'), None, [255, 255, 50]],
            'arm_left': [('LShoulder', 'LWrist'), None, [255, 128, 0]],
            'leg_right': [('RHip', 'RAnkle'), None, [0, 204, 204]],
            'leg_left': [('LHip', 'LAnkle'), None, [0, 204, 204]],
        }

        self.keypoint_columns = [
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
        self.scale_factor = scale_factor

        self.image_h = image_h

        self.image_w = image_w

        self.centers = {}

        self.body_pred = {}

        dirpath = os.getcwd()
        # TODO: find pictures for all robot parts
        self.robot_parts = {'arm_right': {'img': cv2.imread(os.path.join(dirpath, "robot_arm.png"), -1),
                                          'centers': np.array([600, 0])
                                          },  # vector in length of arm
                            'arm_left': {'img': cv2.imread(os.path.join(dirpath, "robot_arm.png"), -1),
                                         'centers': np.array([-600, 0])
                                         },  # vector in length of arm
                            'leg_right': {'img': cv2.imread(os.path.join(dirpath, "robot_arm.png"), -1),
                                          'centers': np.array([600, 0])
                                          },  # vector in length of arm
                            'leg_left': {'img': cv2.imread(os.path.join(dirpath, "robot_arm.png"), -1),
                                         'centers': np.array([-600, 0])
                                         },  # vector in length of arm
                            'face': {'img': cv2.imread(os.path.join(dirpath, "roboy_face.jpeg"), -1),
                                     'centers': np.array([0, 449])
                                     },
                            }

    def get_body_predictions(self, y_pred):

        body_pred = {}

        for j, keypoint in enumerate(self.keypoint_columns):
            test = np.sum(y_pred[:, :, j])
            # If no keypoint was detected heatmap is 0
            if np.sum(y_pred[:, :, j]):
                body_pred[keypoint] = np.unravel_index(y_pred[:, :, j].argmax(), y_pred[:, :, j].shape)
                # print ("body_pred: ", body_pred[keypoint], j)

        return body_pred

    def get_keypoints(self, y_pred):
        body_pred = self.get_body_predictions(y_pred)

        centers = {}
        # draw point
        for key in self.keypoint_columns:
            if key not in body_pred.keys():
                continue

            body_part = body_pred[key]
            center = (body_part[1] * self.scale_factor, body_part[0] * self.scale_factor)
            centers[key] = center

        return centers, body_pred

    def draw_keypoints(self, y_pred, img=None):
        centers, body_pred = self.get_keypoints(y_pred)

        if img is None:
            img = np.zeros((self.image_h*self.scale_factor, self.image_w*self.scale_factor, 3), np.uint8)

        # draw points
        for key in self.keypoint_columns:
            if key not in centers.keys():
                continue

            center = centers[key]
            img[center[1], center[0], :] = [0, 255, 0]

        return img, centers, body_pred

    def plot_stick_man(self, y_pred):

        img, centers, body_pred = self.draw_keypoints(y_pred)

        # draw line
        for pair_order, pair in enumerate(self.keypoint_pairs):
            if pair[0][0] not in body_pred.keys() or pair[0][1] not in body_pred.keys() or pair[1] in body_pred.keys():
                continue

            cv2.line(img, centers[pair[0][0]], centers[pair[0][1]], pair[2], self.scale_factor)

        return img

    def plot_robot_man(self, y_pred, img=None):

        img, centers, body_pred = self.draw_keypoints(y_pred, img)

        # draw line
        for key, pair in self.robot_pairs.items():
            if pair[0][0] not in body_pred.keys() or pair[0][1] not in body_pred.keys() or pair[1] in body_pred.keys():
                continue

            # Get body part
            robot_part = self.robot_parts[key]  # [:,:,:3]
            s_img = robot_part['img']
            s_vec = robot_part['centers']

            # inner product between keypoints vector and img
            keypoint_vector = np.array([centers[pair[0][0]][0] - centers[pair[0][1]][0],
                                        centers[pair[0][0]][1] - centers[pair[0][1]][1]])
            inner_product = np.dot(keypoint_vector/np.linalg.norm(keypoint_vector), s_vec/np.linalg.norm(s_vec))
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
            img_x1 = np.minimum(centers[pair[0][1]][0]+keypoint_vector[0], centers[pair[0][1]][0]-keypoint_vector[0])
            img_x2 = np.maximum(centers[pair[0][1]][0]+keypoint_vector[0], centers[pair[0][1]][0]-keypoint_vector[0])
            img_y1 = np.minimum(centers[pair[0][1]][1]+keypoint_vector[1], centers[pair[0][1]][1]-keypoint_vector[1])
            img_y2 = np.maximum(centers[pair[0][1]][1]+keypoint_vector[1], centers[pair[0][1]][1]-keypoint_vector[1])

            x_avg = np.mean([img_x1, img_x2])
            y_avg = np.mean([img_y1, img_y2])

            y1, y2 = int(y_avg - s_img.shape[0] / 2), int(y_avg + s_img.shape[0] / 2)
            x1, x2 = int(x_avg - s_img.shape[1] / 2), int(x_avg + s_img.shape[1] / 2)

            img[y1:y2, x1:x2, :] = s_img[:, :, :3]

        return img

if __name__ == "__main__":
    plt.interactive(False)
    plotter = Plotter(scale_factor=10)
    y_predictions = np.array(pd.read_pickle('../heatmaps.pkl'))

    #### Iterate over heatmaps in frame
    for y_pred in y_predictions:
        fig = plt.figure()

        centers, body_pred = plotter.get_keypoints(y_pred)
        img = plotter.plot_robot_man(y_pred)
        plt.imshow(img)
        plt.show()