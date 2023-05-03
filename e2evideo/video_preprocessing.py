"""
Video pre-processing module that extract frames from videos.
Main functionality:
    * Create './images' folder, if it does not exist already.
    * Read all videos from a given videos folder and a specified video extension (default mp4).
    * Choose between either Saving all frames or one frame per second,
        in a video in corresponding images folder with the same video file name.
"""
# import packages needed
import argparse
import os
import glob
import math
import cv2
import numpy as np

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
    create_folder(options.output_folder)
    search_path = os.path.join(options.videos_folder, '**', '*' + options.video_format)
    video_files = glob.glob(search_path, recursive=True)
    #video_files = glob.glob(options.videos_folder + "/**/*." + options.video_format,
            #                recursive=True)
    assert len(video_files) != 0 , 'The given videos folder does not contain any vidoes'

    for video_file in video_files:
        print('\n\n')
        video_format_len = len(options.video_format) + 1
        images_sub_folder = video_file.split('/')[-1][:-video_format_len]
        frames_folder = os.path.join(options.output_folder, images_sub_folder)
        create_folder(frames_folder)
        # capture the video from the video file
        cap = cv2.VideoCapture(video_file) # pylint: disable=E1101
        frame_rate = cap.get(cv2.CAP_PROP_FPS) # pylint: disable=E1101
        total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # pylint: disable=E1101
        video_length = total_num_frames / frame_rate
        n_duration = video_length / options.num_frames
        if options.num_frames > total_num_frames:
            #print(f"Warning: the number of frames is larger than the total number of frames in the video {video_file}")
            frame_indices = [i for i in range(int(total_num_frames))]
        else:
            frame_indices = [round(frame_rate * i) for i in range(options.num_frames)]
        count = 0
        frames = []
        while cap.isOpened():
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES) # pylint: disable=E1101
            ret, frame = cap.read()
            if not ret:
                break
            file_name = f"frame{count}." + options.image_format
            
            if options.sampling_mode == 'every_frame':
                cv2.imwrite(frames_folder + '/' + file_name, frame) # pylint: disable=E1101
                count += 1
            elif options.sampling_mode == 'per_second':
                if frame_id % math.floor(frame_rate) == 0:
                    cv2.imwrite(frames_folder + '/'  + file_name, frame) # pylint: disable=E1101
                    count += 1
            elif options.sampling_mode == 'fixed_frames':
                if frame_id in frame_indices:
                    cv2.imwrite(frames_folder + '/' + file_name, frame)
                    count += 1
                    frames.append(frame)
        
            # interpolate missing frames if count is less than the number of frames
            if frame_id == total_num_frames - 1 and options.sampling_mode == 'fixed_frames' and len(frames) < options.num_frames:
                print(f"Warning: the number of frames extracted from {video_file} is less than the number of frames specified")
                print(f"Interpolating missing frames...")
                while len(frames) < options.num_frames:
                    alpha = float(len(frames)) / len(frame_indices)
                    interpolated_frame = cv2.addWeighted(frame, 1 - alpha, frames[-1], alpha, 0)
                    frames.append(interpolated_frame)
                    file_name = f"frame{count}." + options.image_format
                    cv2.imwrite(frames_folder + '/' + file_name, interpolated_frame)
                    count += 1
        cap.release()
        print(f"Done! {count} images of format JPG is saved in {frames_folder}" )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_folder', required=True, help ='Path to the videos folder.')
    parser.add_argument('--video_format', default='mp4', help='choose the video format to read.')
    parser.add_argument('--image_format', default='jpg',
                        help='choose the format for the output images.')
    parser.add_argument('--sampling_mode', default='every_frame', choices= ['fixed_frames', 'every_frame', 'per_second'])
    parser.add_argument('--num_frames', default=10, type=int)
    parser.add_argument('--output_folder', default='../data/ucf_sports_actions/frames')
    opts = parser.parse_args()
    main(opts)
