import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import os
import pandas as pd
import numpy as np
from cv2 import destroyAllWindows, VideoCapture, imshow

if os.name == 'nt':
    from msvcrt import getch, kbhit
else:
    import curses

matplotlib.use('tkagg')

import WalabotAPI as wb

def plotpointcloud(RadarRawImage, ax):
    img_shape = np.shape(RadarRawImage)
    pointrep = list()
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for q in range(img_shape[2]):
                if RadarRawImage[i][j][q] > 0:
                    pointrep.append(np.array([i, j, q]))

    for i in pointrep:
        ax.scatter(i[0] / 4, i[1] / 2, i[2] / 2)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def run():


    cap = VideoCapture(0)

    # Select scan arena
    #             R             Phi          Theta
    ARENA = [(40, 500, 4), (-60, 60, 5), (-15, 15, 5)]

    # Init of Dataframe
    dataset = pd.DataFrame()
    # Star Walabot capture process
    print("Initialize API")
    wb.Init()
    wb.Initialize()

    # Check if a Walabot is connected
    try:
        wb.ConnectAny()

    except wb.WalabotError as err:
        print("Failed to connect to Walabot.\nerror code: " + str(err.code))
        print(wb.GetExtendedError())
        print(wb.GetErrorString())
        sys.exit(1)

    ver = wb.GetVersion()
    print("Walabot API version: {}".format(ver))

    print("Connected to Walabot")
    wb.SetProfile(wb.PROF_SENSOR)

    # Set scan arena
    wb.SetArenaR(*ARENA[0])
    wb.SetArenaPhi(*ARENA[1])
    wb.SetArenaTheta(*ARENA[2])
    print("Arena set")

    # Set image filter
    wb.SetDynamicImageFilter(wb.FILTER_TYPE_NONE)

    # Start calibration
    wb.Start()
    wb.StartCalibration()
    while wb.GetStatus()[0] == wb.STATUS_CALIBRATING:
        wb.Trigger()

    print("Calibration done!")

    namevar = time.strftime("%Y_%m_%d_%H_%M_%S")

    dim1, dim2 = wb.GetRawImageSlice()[1:3]
    try:
        pairs = wb.GetAntennaPairs()
        j = 1
        # print(len(wb.GetAntennaPairs()))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        while True:
            # One iteration takes around 0.1 seconds
            wb.Trigger()
            ret, frame = cap.read()

            try:

                raw_image, size_X, size_Y, size_Z, power = wb.GetRawImage()
                raw_imageSlice, size_phi, size_r, slice_depth, powerSlice = wb.GetRawImageSlice()

                sample_dict = {
                    'timestamp': time.time_ns(),
                    # 'raw_signals': raw_signals,
                    'wlb/img': raw_image,
                    'wlb/slice': raw_imageSlice,
                    'wlb/X': size_X,
                    'wlb/Y': size_Y,
                    'wlb/Z': size_Z,
                    'wlb/power': power,
                    'wlb/Sphi': size_phi,
                    'wlb/Sr': size_r,
                    'wlb/Sdepth': slice_depth,
                    'wlb/Spower': powerSlice,
                    'cam/img': frame
                }
                # plotpointcloud(raw_image, ax)
                # if j%10 == 0:
                #     plt.show()

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                dataset = dataset.append(sample_dict, ignore_index=True)
                # print(sample_dict["timestamp"])
            except:
                print(sys.exc_info())

            j = j+1
            if j%1000==0:
                print("Saving at j=" + str(j))
                dataset.to_pickle("walabot_{}_{}.pkl.zip".format(namevar, int(j/1000)), compression='zip')
                print("Saved!")
                dataset = pd.DataFrame()

    except KeyboardInterrupt:
        print('interrupted!')
    finally:
        dataset.iloc[-1000:].to_pickle("walabot_{}_{}.pkl.zip".format(namevar, int(j/1000)+1), compression='zip')
        cap.release()
        destroyAllWindows()

        wb.Stop()
        wb.Disconnect()

        print("Done!")

    sys.exit(0)

if __name__ == '__main__':
    run()