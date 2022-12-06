# import packages needed
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import os 

import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
import imutils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

count = 0 
video_file = './data/DataSet1/Videos/2020-06-26_18-28-10_camera102.mp4'
# capture the video from the video file
cap = cv2.VideoCapture(video_file)
frame_rate = cap.get(5)
x = 1 
while (cap.isOpened()):
    frame_id = cap.get(1)
    ret, frame = cap.read()
    if(ret != True):
        break
    if (frame_id % math.floor(frame_rate) == 0):
        file_name = 'frame%d.jpg' % count 
        count += 1
        #cv2.imwrite(file_name, frame)
cap.release()
print('Done! Number of files saved is {:d} in the format of '.format(count) )
