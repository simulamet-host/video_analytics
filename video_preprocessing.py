# import packages needed
import argparse

import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import os 
import glob

import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
import imutils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--videos_folder', required=True, help ='Path to the videos folder.')
parser.add_argument('--video_format', default='mp4')
parser.add_argument('--image_format', default='jpg')
options = parser.parse_args()

try:
    os.makedirs('./images')
except OSError:
    pass

video_files = glob.glob(options.videos_folder + "*." + options.video_format)

for video_file in video_files:
    count = 0
    folder_name_len = len(options.videos_folder)
    video_format_len = len(options.video_format) + 1
    images_folder = './images/' + video_file[folder_name_len:-video_format_len ]

    try:
        os.makedirs(images_folder)
    except OSError:
        pass

    # capture the video from the video file
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    x = 1 
    while (cap.isOpened()):
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if(ret != True):
            break
        if (frame_id % math.floor(frame_rate) == 0):
            file_name = 'frame%d.' % count + options.image_format
            count += 1
            cv2.imwrite(images_folder + '/' + file_name, frame)
    cap.release()
    print('\nDone! {:d} images of format JPG is saved in {} \n'.format(count, images_folder) )
