#%%
import os
import argparse
from image_preprocessing import get_images

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns
import cv2
import random
# %%
opt_dict = {'dir':  '../data/ucf_sports_actions/frames', 'img_format': '*.jpg', 'resize': True, 'img_width': 60, 'img_height': 60, 'gray_scale': False, 'output': 'all_images.npy', 
            'output': './results/ucf_sports_actions_images.npz'}
opt_ = argparse.Namespace(**opt_dict)
video_array, file_list = get_images(opt_)
# %%
file_list
# %%
labels_dict = {
    'Diving': 0,
    'GolfSwing': 1,
    'Kicking': 2,
    'Lifting': 3,
    'RidingHorse': 4,
    'Running': 5,
    'SkateBoarding': 6,
    'SwingBench': 7,
    'SwingSide': 8,
    'Walking': 9
}
#%%
def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key

# %%
def get_activity_name(file_name):
    name_parts = file_name.split('_')
    return ' '.join(name_parts[1:-1])

labels = [labels_dict[get_activity_name(file)] for file in file_list]

print(labels)
#%%
# Choose a random video index and frame index
random_video_index = [random.randint(0, video_array.shape[0] - 1) for _ in range(8)]
random_frame_index = [random.randint(0, video_array.shape[1] - 1) for _ in range(8)]

# Display the random frames using matplotlib
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()
# Get the random frame
for i, (video_index, frame_index) in enumerate(zip(random_video_index, random_frame_index)):
    ax = axes[i]
    random_frame = video_array[video_index, frame_index]
    resized_frame = cv2.resize(random_frame , (224, 224)) 
    random_frame_label = labels[video_index]
    # get the corresponding label from the label dictionary
    frame_key = get_key_by_value(labels_dict, random_frame_label)
    ax.imshow(resized_frame)
    ax.set_title(frame_key, fontsize = 20, color= 'purple')
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
