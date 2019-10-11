import cv2
import errno
import glob
import pickle

import numpy as np
import pandas as pd


def get_raw_data(df_row):
    rostime = df_row.rostime
    topic = df_row.topic
    data = df_row.raw

    if topic == "/camera/color/image_raw/compressed":
        nparr = np.fromstring(data[1], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return rostime, img

    elif topic == '/camera/depth/image_rect_raw/compressed':
        nparr = np.fromstring(data[1], np.uint8)
        img = cv2.imdecode(nparr, 2)
        return rostime, img

    elif topic == '/p2g_base_scans':
        return rostime, data

    elif topic == '/ts_scans':
        return rostime, data

    else:
        return None;


def main():
    paths = ["/home/roboy/Projects/RoboyMedium/data/unlabeled/20190826_openspace/"]

    # ## Load data frames
    for path in paths:

        files = glob.glob(path + "*.pkl")
        files.sort()
        rosbag_files = [file for file in files if not file.split('/')[-1].startswith('walabot')]
        walabot_files = [file for file in files if file.split('/')[-1].startswith('walabot')]

        # In[ ]:

        search_idx = {
            'p2g_0': 0,
            'p2g_1': 0,
            'p2g_2': 0,
            'p2g_3': 0,
            'ts3_0': 0,
            'ts3_1': 0,
            'frame': 0
        }

        search_range = 10

        # In[ ]:

        rosbag_files

        # In[ ]:

        rostimes = []

        out = cv2.VideoWriter(path + 'raw_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (640, 480))

        dataset = pd.DataFrame()

        for i, name in enumerate(rosbag_files):

            begin_idx = dataset.shape[0]

            filename = name.split('/')[-1].split('.')[0]

            print("Start processing {}".format(filename))

            try:
                df = pd.read_pickle(name).sort_values(by='rostime')

                search_idx = {
                    'p2g_0': 0,
                    'p2g_1': 0,
                    'p2g_2': 0,
                    'p2g_3': 0,
                    'ts3_0': 0,
                    'ts3_1': 0,
                    'frame': 0
                }

                frames = df[df['topic'] == "/camera/color/image_raw/compressed"]

                # The frames are hardcoded in order to deal with missing data, i.e. that
                # sensor locations are consistent over all measurements even if a sensor
                # was missing in one dataset
                p2g_scans = {}
                p2g_scans['p2g_0'] = df[df.frame == 'p2g_0']
                p2g_scans['p2g_1'] = df[df.frame == 'p2g_1']
                p2g_scans['p2g_2'] = df[df.frame == 'p2g_2']
                p2g_scans['p2g_3'] = df[df.frame == 'p2g_3']

                ts3_scans = {}
                ts3_scans['ts3_0'] = df[df.frame == 'ts3_0']
                ts3_scans['ts3_1'] = df[df.frame == 'ts3_1']

                for index, p2g_0_scan in p2g_scans['p2g_0'].iterrows():

                    rostime, _ = get_raw_data(p2g_0_scan)

                    # Build sample dict
                    sample_dict = {'filename': filename,
                                   'rostime': rostime,
                                   }

                    # get frame with rostime closest to current radar scan
                    while frames.iloc[search_idx['frame']].rostime - rostime < 0:
                        search_idx['frame'] = search_idx['frame'] + 1

                    if (frames.iloc[search_idx['frame']].rostime - rostime) > 10e9:
                        print("No camera frame within one second found. Skipping!")
                        print(frames.iloc[search_idx['frame']].rostime - rostime)
                        continue

                    frame = frames.iloc[search_idx['frame']]
                    frame_raw = np.frombuffer(frame.raw[1], np.uint8)
                    frame_img = cv2.imdecode(frame_raw, cv2.IMREAD_COLOR)

                    out.write(frame_img)

                    rostimes.append(float(rostime))

                    # get ultrasonic targets closest to current radar scan
                    for id in range(0, 2):
                        key = 'ts3_{}'.format(id)
                        try:
                            while ts3_scans[key].iloc[search_idx[key]].rostime - rostime < 0:
                                search_idx[key] = search_idx[key] + 1

                            if (ts3_scans[key].iloc[search_idx[key]].rostime - rostime) > 10e9:
                                print("No ts3 frame within one second found. Skipping!")
                                sample_dict[key] = ts3_scans[key].iloc[search_idx[key]].raw
                            else:
                                sample_dict[key] = []
                        except IndexError:
                            sample_dict[key] = []

                    # get other p2g antennas closest to current radar scan
                    for id in range(0, 4):
                        key = 'p2g_{}'.format(id)
                        try:
                            while p2g_scans[key].iloc[search_idx[key]].rostime - rostime < 0:
                                search_idx[key] = search_idx[key] + 1
                            if search_idx[key] >= p2g_scans[key].shape[0]:
                                raise Exception("Time out exeption!")

                            if (p2g_scans[key].iloc[search_idx[key]].rostime - rostime) > 10e9:
                                print("No {} frame within one second found. Skipping!".format(key))
                                raise Exception("Time out exeption!")

                            _, frame = get_raw_data(p2g_scans[key].iloc[search_idx[key]])
                            sample_dict[key + "_0real"] = frame['antenna0real']
                            sample_dict[key + "_0imag"] = frame['antenna0imag']
                            sample_dict[key + "_1real"] = frame['antenna1real']
                            sample_dict[key + "_1imag"] = frame['antenna1imag']
                        except:
                            sample_dict[key + "_0real"] = []
                            sample_dict[key + "_0imag"] = []
                            sample_dict[key + "_1real"] = []
                            sample_dict[key + "_1imag"] = []

                    # append to dataset
                    dataset = dataset.append(sample_dict, ignore_index=True)

                print('Finished processing {}'.format(filename))

            except IOError as exc:
                if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                    print(exc)  # Propagate other kinds of IOError.
            except IndexError as exc:
                print(exc)

        dataset.to_pickle(path + "dataset_{}.pkl".format(filename))
        print('\n Overall duration: {}'.format((dataset.iloc[-1].rostime - dataset.iloc[0].rostime) / 10e8))

        out.release()  # Finish supervision video

        # In[ ]:

        # Store list of rostimes per video frame
        with open(path + 'rostimes_synchronized.pkl', 'wb') as f:
            pickle.dump(rostimes, f)

        dataset.to_pickle(path + "synchronized_{}.pkl".format(path.split("/")[-2]))

        # ### Sync walabot dataframes into

        total_sum = 0
        walabot_df = pd.DataFrame()
        for walabot_file in walabot_files:
            print(walabot_file)
            walabot_df = walabot_df.append(pd.read_pickle(walabot_file))
            print("shape: ", walabot_df.shape)

        if 'timestamp' in walabot_df.columns:
            dataset = dataset.sort_values(by=['rostime'])
            walabot_df = walabot_df.sort_values(by=['timestamp'])

            data_merged = pd.merge_asof(left=dataset, left_on='rostime',
                                        right=walabot_df, right_on='timestamp',
                                        tolerance=5 * 10e8
                                        )

            data_merged.to_pickle(path + "merged_{}.pkl".format(path.split("/")[-2]))


if __name__ == '__main__':
    main()
