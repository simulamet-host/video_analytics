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
    all_images = []
    for folder_name in img_folders:
        images_ = glob.glob(folder_name + "/"+ args_opt.img_format )
        for img_ in images_:
            img = cv2.imread(img_) # pylint: disable=E1101
            if args_opt.gray_scale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # pylint: disable=E1101
            if args_opt.resize:
                img = cv2.resize(img, (args_opt.img_width, args_opt.img_height)) # pylint: disable=E1101
            all_images.append(img_as_float32(img))
    all_images = np.array(all_images)
    return all_images

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--dir', default='./images/')
    parser_.add_argument('--img_format', default='*.jpg')
    parser_.add_argument('--resize', default=False)
    parser_.add_argument('--img_width', default=224, type=int)
    parser_.add_argument('--img_height', default=224, type=int)
    parser_.add_argument('--gray_scale', default=False)
    args_ = parser_.parse_args()
  
    _images = get_images(args_)
  
    print('Images saved in array of arrays with size', str(_images.shape))