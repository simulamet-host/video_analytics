"""
Video pre-processing module that extract frames from videos.

Main functionality:
    * Create './images' folder, if it does not exist already.
    * Read all videos from a given videos folder and a specified video extension (default mp4).
    * Choose between either Saving all frames or one frame per second, in a video in corresponding images folder with the same video file name.

Author Faiga Alawad 2022.
"""
# import packages needed
import argparse
import os
import glob
import math
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

def main(options):
    """
    Main function will run when the python file is called from terminal.
    """
    assert os.path.isdir(options.videos_folder), 'The given videos_folder does not exist'
    create_folder('./images')
    video_files = glob.glob(options.videos_folder + "*." + options.video_format)
    assert len(video_files) != 0 , 'The given videos folder does not contain any vidoes'
    for video_file in video_files:
        count = 0
        folder_name_len = len(options.videos_folder)
        video_format_len = len(options.video_format) + 1
        images_folder = './images/' + video_file[folder_name_len:-video_format_len ]
        create_folder(images_folder)
        
        # capture the video from the video file
        cap = cv2.VideoCapture(video_file) # pylint: disable=E1101
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret:
                break
            file_name = f"frame{count}."+ options.image_format
            count += 1
            if options.how_often == 'all_frames':
                cv2.imwrite(images_folder + '/' + file_name, frame) # pylint: disable=E1101
            elif options.how_often == 'per_second':
                if (frame_id % math.floor(frame_rate) == 0):
                    cv2.imwrite(images_folder + '/'  + file_name, frame)
        cap.release()
        print(f"\nDone! {count} images of format JPG is saved in {images_folder}" )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_folder', required=True, help ='Path to the videos folder.')
    parser.add_argument('--video_format', default='mp4', help='choose the video format to read.')
    parser.add_argument('--image_format', default='jpg',
                        help='choose the format for the output images.')
    parser.add_argument('--how_often', default='all_frames', choices= ['all_frames', 'per_second'])
    opts = parser.parse_args()

    main(opts)
