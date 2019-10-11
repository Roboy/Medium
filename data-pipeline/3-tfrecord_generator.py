# # Generate TFRecords from datasets and supervised poses

# ## Setup
import glob
import h5py
import json
import os
import pandas as pd
import pickle

from utils.tf_records_generator import *

NUM_PERSONS = 2

P2G_TEST_SET_PATH = "/home/kingkolibri/10_catkin_ws/test_records/"
P2G_TRAIN_SET_PATH = "/home/kingkolibri/10_catkin_ws/train_records/"

WLB_TEST_SET_PATH = "/home/kingkolibri/10_catkin_ws/test_hdf5/"
WLB_TRAIN_SET_PATH = "/home/kingkolibri/10_catkin_ws/train_hdf5/"

TOPOSENS_PATH = "/home/kingkolibri/10_catkin_ws/toposens_files/"


def write_to_wlb_hdf(path, filename, input_data, label_data):
    input_shape = (8, 116, 25)
    label_shape = (13, 116, 25)

    try:
        os.remove(path + filename)
    except:
        pass

    # open a hdf5 file and create arrays
    hdf5_file = h5py.File(path + filename, mode='w')

    for i in range(len(input_data)):
        group = hdf5_file.create_group(str(i))
        # changed np.uint to float in raw image
        group.create_dataset("raw_image", input_shape, np.float32, input_data[i])
        group.create_dataset("heatmap", label_shape, np.float32, label_data[i])

    hdf5_file.close()


def gaussian_k(x0, y0, sigma, height, width):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    y = np.arange(0, width, 1, float)
    x = np.arange(0, height, 1, float)[:, np.newaxis]
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


# In[15]:

def generate_heatmap(height, width, landmarks, sigma=3):
    """ Generate a full heat map for every landmark in an array
    """
    num_landmarks = landmarks.shape[0]
    hm = np.zeros((height, width, num_landmarks), dtype=np.float32)
    for i in range(num_landmarks):
        if not np.array_equal(landmarks[i], [-1, -1]):

            hm[:, :, i] = gaussian_k(landmarks[i][0] * height,
                                     landmarks[i][1] * width,
                                     sigma,
                                     height, width)
        else:
            hm[:, :, i] = np.zeros((height, width))

    return hm


def compute_rd_maps(x, window_func='hamming'):
    rd_maps = []

    n = int(2 ** np.ceil(np.log2(x.shape[1])))
    m = int(2 ** np.ceil(np.log2(x.shape[0])))

    if window_func == 'hamming':
        window = np.hanning(x.shape[1])
    else:
        window = np.ones(x.shape[1])

    for radar_matrix in x.T:

        range_matrix = []

        # apply fast time fft
        for chirp in radar_matrix.T:
            range_matrix.append(np.fft.fftshift(np.fft.fft(chirp * window - np.mean(chirp), n=n)))

        range_matrix = np.array(range_matrix)

        # apply slow time fft
        rd_map = []
        for range_bin in range_matrix.T:
            rd_map.append(np.fft.fftshift(np.fft.fft(range_bin, m)))

        rd_maps.append(rd_map)

    return np.array(rd_maps).transpose()


def main():
    paths = ['/home/roboy/Projects/RoboyMedium/data/unlabeled/20190825_BauIngHall/',
             '/home/kingkolibri/10_catkin_ws/20190825_OutsideAudimax/',
             '/home/roboy/Projects/RoboyMedium/data/unlabeled/20190826_openspace/',
             '/home/roboy/Projects/RoboyMedium/data/unlabeled/20190827_openspace1/',
             '/home/roboy/Projects/RoboyMedium/data/unlabeled/20190827_openspace3/',
             '/home/roboy/Projects/RoboyMedium/data/unlabeled/20190827_openspace4/',
             '/home/roboy/Projects/RoboyMedium/data/unlabeled/20190828_openspace0/',
             '/home/roboy/Projects/RoboyMedium/data/unlabeled/20190828_openspace1/'
             ]
    for DATASET_PATH in paths:

        try:

            print(DATASET_PATH)

            POSES_PATH = DATASET_PATH + "results/poses/"

            folder_name = DATASET_PATH.split("/")[-2]
            files = glob.glob(DATASET_PATH + "*.pkl")
            rosbag_files = [file for file in files if file.split('/')[-1].startswith('merged')]
            rosbag_files.sort()

            # Load list of rostimes per video frame
            with open(DATASET_PATH + 'rostimes_synchronized.pkl', 'rb') as f:
                rostimes = pickle.load(f)

            # In[8]:

            keypoint_columns = {
                'Nose': 0,
                'Neck': 1,
                'RShoulder': 2,
                'RElbow': 3,
                'RWrist': 4,
                'LShoulder': 5,
                'LElbow': 6,
                'LWrist': 7,  # LWrist
                'MidHip': 8,
                'RHip': 9,  # RHip
                'RKnee': 10,  # RKnee
                'RAnkle': 11,  # RAnkle
                'LHip': 12,  # LHip
                'LKnee': 13,  # LKnee
                'LAnkle': 14,  # LAnkle
                'REye': 15,  # REye
                'LEye': 16,  # LEye
                'REar': 17,  # REar
                'LEar': 18,  # LEar
                'LBigToe': 19,
                'LSmallToe': 20,
                'LHeel': 21,
                'RBigToe': 22,
                'RSmallToe': 23,
                'RHeel': 24,
                'Background': 25
            }

            keypoint_keys = list(keypoint_columns.keys())

            if os.path.exists(DATASET_PATH + "dataset_supervised.pickle"):
                data = pd.read_pickle(DATASET_PATH + "dataset_supervised.pickle")
            else:
                data = pd.DataFrame()

                for file in rosbag_files:

                    dataset = pd.read_pickle(file)
                    num_rows = dataset.shape[0]

                    for index, row in dataset.iterrows():

                        row_data = row.to_dict()

                        rostime = row_data['rostime']
                        try:
                            with open(POSES_PATH + "/raw_video_{}_keypoints.json".format(
                                    str(rostimes.index(rostime)).zfill(12)), 'r') as file:
                                parsed = json.load(file)
                        except FileNotFoundError as e:
                            break

                        sample_dict = {}

                        num_persons = np.minimum(2, len(parsed['part_candidates']))

                        # print("num persons: {}".format(num_persons))

                        for person in range(0, num_persons):
                            for key in parsed['part_candidates'][person].keys():
                                if parsed['part_candidates'][person][key]:
                                    sample_dict['keypoint_{0}_{1}_x'.format(person, keypoint_keys[int(key)])] = \
                                        parsed['part_candidates'][person][key][0]
                                    sample_dict['keypoint_{0}_{1}_y'.format(person, keypoint_keys[int(key)])] = \
                                        parsed['part_candidates'][person][key][1]
                                    sample_dict['keypoint_{0}_{1}_score'.format(person, keypoint_keys[int(key)])] = \
                                        parsed['part_candidates'][person][key][2]
                                else:
                                    sample_dict['keypoint_{0}_{1}_x'.format(person, keypoint_keys[int(key)])] = None
                                    sample_dict['keypoint_{0}_{1}_y'.format(person, keypoint_keys[int(key)])] = None
                                    sample_dict['keypoint_{0}_{1}_score'.format(person, keypoint_keys[int(key)])] = None

                        row_data.update(sample_dict)

                        data = data.append(row_data, ignore_index=True)

                data.to_pickle(DATASET_PATH + "dataset_supervised.pickle")

            # ### Extract data for toposens
            toposens_columns = [column for column in data.columns
                                if not (column.startswith("p2g") or
                                        column.startswith("timestamp") or
                                        column.startswith("raw"))
                                ]

            data[toposens_columns].to_pickle(TOPOSENS_PATH + "ts3-{0}.pickle.zip".format(folder_name),
                                             compression="zip")

            # ## Keypoints heatmaps

            # Set hyper parameters
            num_pixel_width = 64
            num_pixel_height = 64
            sigma = 5

            # Uncomment keypoints to consider
            landmark_columns = [
                ['keypoint_0_LAnkle_x', 'keypoint_0_LAnkle_y'],
                # ['keypoint_0_LBigToe_x', 'keypoint_0_LBigToe_y'],
                # ['keypoint_0_LEar_x', 'keypoint_0_LEar_y'],
                ['keypoint_0_LElbow_x', 'keypoint_0_LElbow_y'],
                # ['keypoint_0_LEye_x', 'keypoint_0_LEye_y'],
                # ['keypoint_0_LHeel_x', 'keypoint_0_LHeel_y'],
                ['keypoint_0_LHip_x', 'keypoint_0_LHip_y'],
                ['keypoint_0_LKnee_x', 'keypoint_0_LKnee_y'],
                ['keypoint_0_LShoulder_x', 'keypoint_0_LShoulder_y'],
                # ['keypoint_0_LSmallToe_x', 'keypoint_0_LSmallToe_y'],
                ['keypoint_0_LWrist_x', 'keypoint_0_LWrist_y'],
                # ['keypoint_0_MidHip_x', 'keypoint_0_MidHip_y'],
                # ['keypoint_0_Neck_x', 'keypoint_0_Neck_y'],
                ['keypoint_0_Nose_x', 'keypoint_0_Nose_y'],
                ['keypoint_0_RAnkle_x', 'keypoint_0_RAnkle_y'],
                # ['keypoint_0_RBigToe_x','keypoint_0_RBigToe_y'],
                # ['keypoint_0_REar_x','keypoint_0_REar_y'],
                ['keypoint_0_RElbow_x', 'keypoint_0_RElbow_y'],
                # ['keypoint_0_REye_x','keypoint_0_REye_y'],
                # ['keypoint_0_RHeel_x','keypoint_0_RHeel_y'],
                ['keypoint_0_RHip_x', 'keypoint_0_RHip_y'],
                ['keypoint_0_RKnee_x', 'keypoint_0_RKnee_y'],
                ['keypoint_0_RShoulder_x', 'keypoint_0_RShoulder_y'],
                # ['keypoint_0_RSmallToe_x', 'keypoint_0_RSmallToe_y'],
                ['keypoint_0_RWrist_x', 'keypoint_0_RWrist_y'],
                ['keypoint_1_LAnkle_x', 'keypoint_1_LAnkle_y'],
                # ['keypoint_1_LBigToe_x', 'keypoint_1_LBigToe_y'],
                # ['keypoint_1_LEar_x', 'keypoint_1_LEar_y'],
                ['keypoint_1_LElbow_x', 'keypoint_1_LElbow_y'],
                # ['keypoint_1_LEye_x', 'keypoint_1_LEye_y'],
                # ['keypoint_1_LHeel_x', 'keypoint_1_LHeel_y'],
                ['keypoint_1_LHip_x', 'keypoint_1_LHip_y'],
                ['keypoint_1_LKnee_x', 'keypoint_1_LKnee_y'],
                ['keypoint_1_LShoulder_x', 'keypoint_1_LShoulder_y'],
                # ['keypoint_1_LSmallToe_x', 'keypoint_1_LSmallToe_y'],
                ['keypoint_1_LWrist_x', 'keypoint_1_LWrist_y'],
                # ['keypoint_1_MidHip_x', 'keypoint_1_MidHip_y'],
                # ['keypoint_1_Neck_x', 'keypoint_1_Neck_y'],
                ['keypoint_1_Nose_x', 'keypoint_1_Nose_y'],
                ['keypoint_1_RAnkle_x', 'keypoint_1_RAnkle_y'],
                # ['keypoint_1_RBigToe_x','keypoint_1_RBigToe_y'],
                # ['keypoint_1_REar_x','keypoint_1_REar_y'],
                ['keypoint_1_RElbow_x', 'keypoint_1_RElbow_y'],
                # ['keypoint_1_REye_x','keypoint_1_REye_y'],
                # ['keypoint_1_RHeel_x','keypoint_1_RHeel_y'],
                ['keypoint_1_RHip_x', 'keypoint_1_RHip_y'],
                ['keypoint_1_RKnee_x', 'keypoint_1_RKnee_y'],
                ['keypoint_1_RShoulder_x', 'keypoint_1_RShoulder_y'],
                # ['keypoint_1_RSmallToe_x', 'keypoint_1_RSmallToe_y'],
                ['keypoint_1_RWrist_x', 'keypoint_1_RWrist_y']
            ]

            # Set hyperparameters
            num_frames = 8  # = 2seconds windows
            num_chirps = 8

            # # ### Compose radar matrix windows
            # X = []
            #
            # shape = np.array(data.p2g_0_0real.tolist()).shape
            #
            # for board in range(0, 4):
            #     for antenna in range(0, 2):
            #         key = "p2g_{0}_{1}".format(board, antenna)
            #
            #         samples = []
            #
            #         if data[key + "real"][0]:
            #             for row in data[key + "real"]:
            #                 if not row:
            #                     samples.append(-1 * np.ones((1024)))
            #                 else:
            #                     samples.append(row)
            #
            #             samples = np.array(samples)
            #             # print(samples.shape)
            #             # samples = np.array(data[key + "real"].tolist())# + np.asarray(data[key + "imag"].tolist())*1j
            #         else:
            #             samples = -1 * np.ones(shape)
            #
            #         X.append(samples.T)
            #
            # X = np.array(X).T

            X_wlb = []
            for row in data['raw_image/img']:
                if row == row:  # check if is not NaN
                    X_wlb.append(row)

            X_wlb = np.array(X_wlb)
            # ### Compute range-doppler maps

            # ## Generate TFRecords
            number_of_elements = int(data.shape[0] - num_frames)

            # Prepare writers for TFRecords
            # p2g_writer = tf.python_io.TFRecordWriter(P2G_TRAIN_SET_PATH + "/p2g_{0}.tfrecord".format(folder_name))
            # ts3_writer = tf.python_io.TFRecordWriter(P2G_TRAIN_SET_PATH + "/ts3_{0}.tfrecord".format(folder_name))
            #
            # test_set_break = int(number_of_elements * 0.85) + int(number_of_elements * 0.85) % 2
            #
            # p2g_inputs = []
            # p2g_labels = []
            #
            # for i in range(0, number_of_elements, 2):
            #
            #     row = data.iloc[i + num_frames]
            #
            #     # Generate heatmaps
            #     landmarks = []
            #     for cols in landmark_columns:
            #         if cols[0] in row.keys():
            #             try:
            #                 landmarks.append([row[cols[0]] / 640, row[cols[1]] / 480])
            #             except TypeError:
            #                 landmarks.append([-1, -1])
            #
            #     landmarks = np.array(landmarks)
            #
            #     heatmaps = generate_heatmap(num_pixel_height,
            #                                 num_pixel_width,
            #                                 landmarks,
            #                                 sigma=sigma
            #                                 )
            #
            #     # TS3 data
            #     ts3_writer.write(generate_ts3_feature(ts3_points=[row['ts3_0'], row['ts3_1']],
            #                                           heatmaps=heatmaps,
            #                                           filename=row['filename']
            #                                           ).SerializeToString())
            #
            #     # P2G data
            #     # Compose radar matrix
            #     X_raw = np.reshape(X[i:i + num_frames:2], (num_frames * num_chirps, 64, 8), 'C')
            #
            #     # Compute rd-map
            #     X_rdm = compute_rd_maps(X_raw)
            #
            #     p2g_inputs.append(X_rdm)
            #     p2g_labels.append(heatmaps)
            #
            #     p2g_writer.write(generate_p2g_feature(X_rdm=X_rdm,
            #                                           heatmaps=heatmaps,
            #                                           filename=row['filename']
            #                                           ).SerializeToString())
            #
            #     # Last 15% of each dataset is going to build our test set
            #     if i == test_set_break:
            #         print("Starting test set")
            #
            #         p2g_writer.close()
            #         ts3_writer.close()
            #
            #         p2g_writer = tf.python_io.TFRecordWriter(
            #             P2G_TEST_SET_PATH + "/p2g_{0}.tfrecord".format(folder_name))
            #         ts3_writer = tf.python_io.TFRecordWriter(
            #             P2G_TEST_SET_PATH + "/ts3_{0}.tfrecord".format(folder_name))
            #
            # p2g_writer.close()
            # ts3_writer.close()

            test_set_break = int(number_of_elements * 0.85) + int(number_of_elements * 0.85) % 2
            path = WLB_TRAIN_SET_PATH

            wlb_inputs = []
            wlb_labels = []

            try_counter = 0

            for i in range(0, number_of_elements, 2):

                # Walabot data
                if 'raw_image/img' in row.keys() and i + num_frames < X_wlb.shape[0]:  # no walabot data for this one

                    row = data.iloc[i + num_frames]

                    # Generate heatmaps
                    landmarks = []
                    missing_count = 0
                    for cols in landmark_columns:
                        if cols[0] in row.keys():
                            try:
                                landmarks.append([row[cols[0]] / 640, row[cols[1]] / 480])
                            except TypeError:
                                landmarks.append([-1, -1])
                                missing_count = missing_count + 1
                    if missing_count < 3:
                        landmarks = np.array(landmarks)

                        heatmaps_wlb = generate_heatmap(25,
                                                        116,
                                                        landmarks,
                                                        sigma=sigma
                                                        )

                        wlb_inputs.append(X_wlb[i:i + num_frames].astype(np.uint8))
                        wlb_labels.append(heatmaps_wlb.T.astype(np.float32))

                        if len(wlb_inputs) > 1000:
                            write_to_wlb_hdf(path, "wlb_{}_{}.hdf5".format(folder_name, int(i / 1000)),
                                             wlb_inputs,
                                             wlb_labels)
                            wlb_inputs = []
                            wlb_labels = []

                    # Last 15% of each dataset is going to build our test set
                    if i == test_set_break:
                        write_to_wlb_hdf(path, "wlb_{}_{}.hdf5".format(folder_name, 1 + int(i / 1000)),
                                         wlb_inputs,
                                         wlb_labels)
                        path = WLB_TEST_SET_PATH

                        wlb_inputs = []
                        wlb_labels = []

            write_to_wlb_hdf(path, "wlb_{}_{}.hdf5".format(folder_name, 'end'), wlb_inputs, wlb_labels)

        except Exception as e:
            print("ERROR!!")
            print(e)
            pass


if __name__ == '__main__':
    main()
