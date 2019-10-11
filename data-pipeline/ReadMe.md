# Data preprocessing pipeline
This repository holds all the steps to populate the TFRecord data loader from pandas dataframes holding the raw ros messages.

## Data supervision pipeline steps

1. Synchronize raw data
    - Notebook: 1-synchronize_raw_data.ipynb  
    - Input files: 
        - {filename}.pkl (outputfile of rosbag_to_pandas)
        - walabot_{filename}.pkl (pickled pandas data frame of walabot data)
     - Output files:
        - dataset_{filename}.pkl 
        - raw_video.avi
        - rostimes_synchronized.pkl  
2. Supervise raw_video.avi
    - Notebook: -
    - Instructions: [guide on conflucence](https://devanthro.atlassian.net/wiki/spaces/SS19/pages/528449656/Install+and+use+OpenPose+docker+image)
    - Input files:
        - raw_video.avi
    - Output files:
        - openpose.avi
        - poses folder containing json files of keypoints
            - I found it fastest to zip the pose folder before copying to a hard drive
3.  Supervise data and generate TFrecords
    - Noteobok: 3-tfrecord_generator.ipynb
    - Input files:
        - dataset_{filename}.pkl 
        - poses folder containing json files of keypoints
        - rostimes_synchronized.pkl  
    - Output files:
        - p2g.tfrecord
        - ts3.tfrecord
