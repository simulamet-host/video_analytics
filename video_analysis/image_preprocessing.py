"""
Image preprocessing module.
Main functionality:
    * Resize images according to a given input dimensions.
"""
import os
import argparse
import glob
import numpy as np
import cv2
from skimage.transform import resize

def get_images(dir_, img_format, re_size, resize_dim):
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
    img_folders = [x[0] for x in os.walk(dir_)]
    all_images = []
    for folder_name in img_folders:
        images_ = glob.glob(folder_name + "/"+ img_format )
        for img_ in images_:
            img = cv2.imread(img_) # pylint: disable=E1101
            if re_size:
                img = resize(img, preserve_range=True, output_shape=resize_dim)
            all_images.append(img)
    all_images = np.array(all_images).astype(np.float32)
    return all_images

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--dir', default='./images/')
    parser_.add_argument('--img_format', default='*.jpg')
    parser_.add_argument('--is_resize', default=True)
    parser_.add_argument('--resize_dim', default=(224, 224))
    args_ = parser_.parse_args()

    _images = get_images(args_.dir, args_.img_format, args_.is_resize, args_.resize_dim)
    print('Images saved in array of arrays with size ', _images.shape)
