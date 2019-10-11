import glob
import os

from utils.plotter import *


def main():
    DATASET_PATH = "/home/roboy/Projects/RoboyMedium/data/20190831_conference_room/"
    POSES_PATH = DATASET_PATH + "results/poses/"

    # In[3]:

    WLB_TEST_SET_PATH = "/home/roboy/Projects/RoboyMedium/data/test_hdf5_small/"
    WLB_TRAIN_SET_PATH = "/home/roboy/Projects/RoboyMedium/data/train_hdf5_small/"

    # In[4]:

    folder_name = DATASET_PATH.split("/")[-2]

    files = glob.glob(DATASET_PATH + "*.pkl")
    dataset_files = [file for file in files if file.split('/')[-1].startswith('dataset')]
    dataset_files.sort()

    # ## Supervise dataframe

    # Load list of rostimes per video frame
    import pickle
    with open(DATASET_PATH + 'raw_video_timestamps.pkl', 'rb') as f:
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

    import json

    data = pd.DataFrame()

    for file in dataset_files:

        dataset = pd.read_pickle(file)
        num_rows = dataset.shape[0]
        print(num_rows)

        for index, row in dataset.iloc[3000:10000].iterrows():

            row_data = row.to_dict()

            rostime = row_data['timestamp']

            with open(POSES_PATH + "/raw_video_{}_keypoints.json".format(str(rostimes.index(rostime)).zfill(12)),
                      'r') as file:
                parsed = json.load(file)

            sample_dict = {}

            num_persons = np.minimum(2, len(parsed['part_candidates']))

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

    # ## Keypoints heatmaps

    # In[ ]:

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
        # ['keypoint_1_LAnkle_x', 'keypoint_1_LAnkle_y'],
        # # ['keypoint_1_LBigToe_x', 'keypoint_1_LBigToe_y'],
        # # ['keypoint_1_LEar_x', 'keypoint_1_LEar_y'],
        # ['keypoint_1_LElbow_x', 'keypoint_1_LElbow_y'],
        # # ['keypoint_1_LEye_x', 'keypoint_1_LEye_y'],
        # # ['keypoint_1_LHeel_x', 'keypoint_1_LHeel_y'],
        # ['keypoint_1_LHip_x', 'keypoint_1_LHip_y'],
        # ['keypoint_1_LKnee_x', 'keypoint_1_LKnee_y'],
        # ['keypoint_1_LShoulder_x', 'keypoint_1_LShoulder_y'],
        # # ['keypoint_1_LSmallToe_x', 'keypoint_1_LSmallToe_y'],
        # ['keypoint_1_LWrist_x', 'keypoint_1_LWrist_y'],
        # # ['keypoint_1_MidHip_x', 'keypoint_1_MidHip_y'],
        # # ['keypoint_1_Neck_x', 'keypoint_1_Neck_y'],
        # ['keypoint_1_Nose_x', 'keypoint_1_Nose_y'],
        # ['keypoint_1_RAnkle_x', 'keypoint_1_RAnkle_y'],
        # # ['keypoint_1_RBigToe_x','keypoint_1_RBigToe_y'],
        # # ['keypoint_1_REar_x','keypoint_1_REar_y'],
        # ['keypoint_1_RElbow_x', 'keypoint_1_RElbow_y'],
        # # ['keypoint_1_REye_x','keypoint_1_REye_y'],
        # # ['keypoint_1_RHeel_x','keypoint_1_RHeel_y'],
        # ['keypoint_1_RHip_x', 'keypoint_1_RHip_y'],
        # ['keypoint_1_RKnee_x', 'keypoint_1_RKnee_y'],
        # ['keypoint_1_RShoulder_x', 'keypoint_1_RShoulder_y'],
        # # ['keypoint_1_RSmallToe_x', 'keypoint_1_RSmallToe_y'],
        # ['keypoint_1_RWrist_x', 'keypoint_1_RWrist_y']
    ]

    def gaussian_k(x0, y0, sigma, height, width):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        y = np.arange(0, width, 1, float)
        x = np.arange(0, height, 1, float)[:, np.newaxis]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def generate_heatmap(height, width, landmarks, sigma=3):
        """ Generate a full heat map for every landmark in an array
        """
        num_landmarks = landmarks.shape[0]
        hm = np.zeros((height, width, num_landmarks), dtype=np.float32)
        for i in range(num_landmarks):
            if not np.array_equal(landmarks[i], [-1, -1]):

                hm[:, :, i] = gaussian_k(landmarks[i][1] * height,
                                         landmarks[i][0] * width,
                                         sigma,
                                         height, width)
            else:
                hm[:, :, i] = np.zeros((height, width))

        return hm

    X_wlb = data['wlb/img'].iloc[1:10].tolist()
    type(X_wlb)

    num_frames = 20

    import h5py

    def write_to_wlb_hdf(path, filename, input_data, label_data):
        input_shape = (len(input_data), num_frames, 7, 25, 116)
        label_shape = (len(input_data), 13, 116, 25)

        try:
            os.remove(path + filename)
        except:
            pass

        # open a hdf5 file and create earrays
        hdf5_file = h5py.File(path + filename, mode='w')
        hdf5_file.create_dataset("raw_image", input_shape, np.uint8)
        hdf5_file.create_dataset("heatmap", label_shape, np.float32)

        for i in range(len(input_data)):
            hdf5_file["raw_image"][i, ...] = input_data[i]
            hdf5_file["heatmap"][i, ...] = label_data[i]

        hdf5_file.close()

        return

    number_of_elements = data.shape[0] - num_frames

    test_set_break = int(number_of_elements * 0.85) + int(number_of_elements * 0.85) % 2
    wlb_inputs = []
    wlb_labels = []
    path = WLB_TRAIN_SET_PATH

    # In[ ]:

    for i in range(number_of_elements):

        row = data.iloc[i + num_frames]

        # Generate heatmaps
        landmarks = []
        for cols in landmark_columns:
            try:
                landmarks.append([row[cols[0]] / 640, row[cols[1]] / 480])
            except TypeError:
                landmarks.append([-1, -1])

        landmarks = np.array(landmarks)

        heatmaps_wlb = generate_heatmap(25, 116,
                                        landmarks,
                                        sigma=5
                                        )

        wlb_inputs.append(np.array(data['wlb/img'].iloc[i:i + num_frames].tolist()).astype(np.uint8))
        #heatmaps_wlb = np.transpose(heatmaps_wlb, (2,0,1))
        wlb_labels.append(heatmaps_wlb.T.astype(np.float32))

        if len(wlb_inputs) > 1000:
            write_to_wlb_hdf(path, "wlb_{}_{}.hdf5".format(folder_name, 80 + int(i / 1000)),
                             wlb_inputs,
                             wlb_labels)
            wlb_inputs = []
            wlb_labels = []

        # Last 15% of each dataset is going to build our test set
        if i == test_set_break:
            write_to_wlb_hdf(path, "wlb_{}_{}.hdf5".format(folder_name, 81 + int(i / 1000)),
                             wlb_inputs,
                             wlb_labels)
            wlb_inputs = []
            wlb_labels = []
            path = WLB_TEST_SET_PATH

    write_to_wlb_hdf(path, "wlb_{}_{}.hdf5".format(folder_name, 82 + int(i / 1000)), wlb_inputs, wlb_labels)


if __name__ == '__main__':
    main()
