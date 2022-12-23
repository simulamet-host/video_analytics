"""
Module for feature extraction from images.
"""
from image_preprocessing import get_images
import matplotlib.pyplot as plt
import cv2



if __name__ == '__main__':
    images_ = get_images('./images/', '*.jpg', True, (224, 224))
    # Show the first image in the directory
    img = images_[0]
    #plt.imshow(img)
    #plt.show()
