#%%
"""
This file create the labels for the action_recognition task from UCF10.
"""
import os
import argparse
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from e2evideo import image_preprocessing
# %%
opt_dict = {'dir':  '../data/ucf_sports_actions/frames/', 'img_format': '*.jpg',
            'resize': True, 'img_width': 60,
            'img_height': 60, 'gray_scale': False,
            'output': './results/ucf_sports_actions_images.npz'}
opt_ = argparse.Namespace(**opt_dict)
video_array, file_list = image_preprocessing.get_images(opt_)
# %%
print(file_list)
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
    """ This function get keys from a dict based on target_values"""
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None
# %%
def get_activity_name(file_name):
    """ This function format the file name to get the activity"""
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
    # pylint: disable=E1101
    resized_frame = cv2.resize(random_frame , (224, 224))
    random_frame_label = labels[video_index]
    # get the corresponding label from the label dictionary
    FRAME_KEY = get_key_by_value(labels_dict, random_frame_label)
    ax.imshow(resized_frame)
    ax.set_title(FRAME_KEY, fontsize = 20, color= 'purple')
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
# Set the path to the parent folder
PARENT_FOLDER = '../data/ucf_sports_actions/frames/'

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["folder", "subfolder_count"])

# Iterate over all subfolders in the parent folder
for folder in os.listdir(PARENT_FOLDER):
    # Check if the current item is a folder
    if os.path.isdir(os.path.join(PARENT_FOLDER, folder)):
        # Get the number of subfolders in the current folder
        subfolder_count = len(os.listdir(os.path.join(PARENT_FOLDER, folder)))
        # Add the results to the DataFrame
        results_df = results_df.append({"folder": folder, "subfolder_count": subfolder_count},
                                       ignore_index=True)

# %%
# Convert the "subfolder_count" column to integer
results_df["subfolder_count"] = results_df["subfolder_count"].astype(int)

# Get the folder with the maximum subfolder count
max_subfolder_folder = results_df.loc[results_df["subfolder_count"].idxmax(), "folder"]

# Get the folder with the minimum subfolder count
min_subfolder_folder = results_df.loc[results_df["subfolder_count"].idxmin(), "folder"]

# Print the folder with the maximum subfolder count
print("Folder with the maximum subfolder count is:", max_subfolder_folder)
# Print the folder with the minimum subfolder count
print("Folder with the minimum subfolder count is:", min_subfolder_folder)

# %%
results_df['subfolder_count'].unique()
# %%
