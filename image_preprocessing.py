"""
Image preprocessing module.
Main functionality: 
    * Resize images according to a given input dimensions.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize


def get_images(dir = './images/', re_size = True, resize_dim = (224, 224)):
    img_folders = [x[0] for x in os.walk(dir)]
    all_images = []
    for folder_name in img_folders[1:]:
        images_ = glob.glob(folder_name + "/"+ "*.jpg" )
        for img_ in images_: 
            img = cv2.imread(img_)
            if re_size:
                img = resize(img, preserve_range=True, output_shape=resize_dim).astype(int)
            all_images.append(img)
    all_images = np.array(all_images)
    return all_images
