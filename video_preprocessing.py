"""
Video pre-processing module that extract frames from videos.

Main functionality:
* Create './images' folder, if it does not exist already.
* Read all videos from a given videos folder and a specified video extension (default mp4).
* Save one frame per second to the images folder.

Author Faiga Alawad 2022.
"""
# import packages needed
import argparse
import os
import math
import glob
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--videos_folder', required=True, help ='Path to the videos folder.')
parser.add_argument('--video_format', default='mp4', help='choose the video format to read.')
parser.add_argument('--image_format', default='jpg',
                    help='choose the format for the output images.')
options = parser.parse_args()

def create_folder(dir_):
    """
    create a folder at dir if it does not exist already.
    Input:
        - dir: the folder directory
    """
    try:
        os.makedirs(dir_)
    except OSError:
        pass

create_folder('./images')

video_files = glob.glob(options.videos_folder + "*." + options.video_format)

for video_file in video_files:
    COUNT = 0
    folder_name_len = len(options.videos_folder)
    video_format_len = len(options.video_format) + 1
    images_folder = './images/' + video_file[folder_name_len:-video_format_len ]
    create_folder(images_folder)

    # capture the video from the video file
    cap = cv2.VideoCapture(video_file) # pylint: disable=E1101
    frame_rate = cap.get(cv2.CAP_PROP_FPS) # pylint: disable=E1101
    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES) # pylint: disable=E1101
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frame_rate) == 0:
            file_name = f"frame{COUNT}."+ options.image_format
            COUNT += 1
            cv2.imwrite(images_folder + '/' + file_name, frame) # pylint: disable=E1101
    cap.release()
    print(f"\nDone! {COUNT} images of format JPG is saved in {images_folder}" )
