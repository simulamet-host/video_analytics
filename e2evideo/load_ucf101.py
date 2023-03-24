import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import utils
from e2evideo import our_utils, plot_results

def load_label(datasets):
    """
    This function is used to load the labels of the dataset.
    """
    label_index=0
    labels=[]
    #Iterate through each foler corresponding to category
    for folder in datasets:
        print(folder)
        for video in tqdm(os.listdir(folder)):
            if video == '.DS_Store':
                print(video)
            else:
                labels.append(label_index)
        label_index+=1
    return np.array(labels, dtype='int8')

def load_ucf101(image_folder, image_array, no_classes = 101, is_load_label = True, is_plot = False):
    if is_load_label:
        label_data = pd.read_csv("../data/UCF-101/ucfTrainTestlist/classInd.txt", sep=' ', header=None, engine="pyarrow")
        label_data.columns=['index', 'labels']
        label_data = label_data.drop(['index'], axis=1)
        label_data = label_data[:no_classes]
        path=[]
        for label in label_data.labels.values:
            # check if the folder in path is not empty
            path_new = image_folder + label+"/"
            path.append(path_new)
    else:
        path = None
        label_data = None

    if is_plot:
        print('\n Plotting the data...\n')
        plot_results.plot_ucf101(label_data)

    print('\n Loading images...\n')
    with np.load(image_array) as images_file:
        images = images_file['arr_0']

    print('\n Loading labels...\n')
    labels_list = load_label(path)

    print('\n \n Shape of images: ', images.shape)
    print('\n \n Shape of labels: ', labels_list.shape)

    #Train Test Split
    print('\n Splitting the data into training and test...\n')
    x_train, x_test, y_train, y_test=train_test_split(images, labels_list, test_size=0.06,
                                                    random_state=10, shuffle=False)

    
    return x_train, x_test, y_train, y_test, label_data
