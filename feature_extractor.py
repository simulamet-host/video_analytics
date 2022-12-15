"""
Module for feature extraction from images.
"""
from image_preprocessing import get_images
import matplotlib.pyplot as plt
import cv2

images_ = get_images('./images/', '*.jpg', True, (224, 224))

# Show the first image in the directory
img = images_[0]
#plt.imshow(img)
#plt.show()

# create SURF object, set Hessian threshold to 400
surf = cv2.SURF_create(400)

# find keypoints and descriptors
kp, des = surf.detectAndCompute(img,None)
print(len(kp))
