"""
Image preprocessing module.
Main functionality:
    * Resize images according to a given input dimensions.
    * Convert images to grayscale.
"""
import os
import argparse
import glob
import numpy as np
import cv2
from skimage import img_as_float32

def get_images(args_opt):
    """
    Read images from a given dir, using specified image format.
    Additionally it allows for resizing the images.
    Parameters:
        - dir_: directory contains the images.
        - img_format: specify the images format, it is string in the format "*.xxx".
        - resize: this is a boolean, set True if resizing of the images are needed.
        - resize_dim: set to the required image dimensions.
                    It takes input in the format (height, width).
    Returns:
        -- all_images: an array of array, it contains all values from the images in the dir.
                        This array is of size = (num_images, height, width, no_channels)
    """
    img_folders = [x[0] for x in os.walk(args_opt.dir)]
    all_videos = []
    for folder_name in img_folders:
        video_file = []
        images_ = glob.glob(folder_name + "/"+ args_opt.img_format )
        for img_ in images_:
            img = cv2.imread(img_) # pylint: disable=E1101
            if args_opt.gray_scale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # pylint: disable=E1101
            if args_opt.resize:
                img = cv2.resize(img, (args_opt.img_width, args_opt.img_height)) # pylint: disable=E1101
            video_file.append(img_as_float32(img))
            
        if len(video_file) !=0:
            # convert video_file to numpy array
            video_file = np.array(video_file)
            all_videos.append(video_file)

    # find the maximum length of the videos (number of frames) in a video
    max_frames = max([len(x) for x in all_videos])
    print(max_frames)
    # find the maximum shape of the arrays
    #max_shape = max([arr.shape for arr in all_images])
    # create a new array with the maximum shape
    # specify the desired shape of the padded arrays
    frame_dim = all_videos[0][0].shape
    frames_in_videos_dim = (len(all_videos), max_frames) + frame_dim

    # generate a list of pad tuples for each array in frames_arr
    pad_tuples = [tuple([(0, max_frames - arr.shape[i]) for i in range(len(arr.shape))]) for arr in all_videos]
    print(pad_tuples)
    # Pad each array in frames_arr with zeros
    padded_arrs = [np.pad(arr, pad_tuple, mode='constant') for arr, pad_tuple in zip(all_videos, pad_tuples)]

    #frames_in_videos = np.zeros(frames_in_videos_dim)
    # pad the shorter videos with zeros at the end to make them all the same length
    """
    for i in range(len(all_videos)):
        frames_arr = all_videos[i]
        if len(frames_arr) < max_frames:
            # generate a list of pad tuples for each array in frames_arr
            pad_tuples = [tuple([(0, max_frames[j] - arr.shape[j]) for j in range(len(arr.shape))]) for arr in frames_arr]
            # pad each array in frames_arr with zeros
            padded_arrs = [np.pad(arr, pad_tuple, mode='constant') for arr, pad_tuple in zip(frames_arr, pad_tuples)]

            #pad_tuple = tuple([(0, max_frames - frames_arr[j].shape) for j in range(len(frames_arr[0].shape))])
            #padded_arr = np.pad(frames_arr, pad_tuple, mode='constant', constant_values=0)
            frames_in_videos[i] = padded_arrs
        else:
            frames_in_videos[i] = frames_arr
    """

    np.save(args_opt.output, padded_arrs)
    return padded_arrs

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--dir', default='./images/')
    parser_.add_argument('--img_format', default='*.jpg')
    parser_.add_argument('--resize', default=False)
    parser_.add_argument('--img_width', default=224, type=int)
    parser_.add_argument('--img_height', default=224, type=int)
    parser_.add_argument('--gray_scale', default=False)
    parser_.add_argument('--output', default='./results/all_images.npy')
    args_ = parser_.parse_args()
    
    _images = get_images(args_)

    print('Images saved in array of array of size', str(_images.shape))
