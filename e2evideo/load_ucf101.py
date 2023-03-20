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
        for _ in tqdm(os.listdir(folder)):
            labels.append(label_index)
        label_index+=1
    return np.array(labels, dtype='int8')

def load_ucf101():
    label_data = pd.read_csv("../data/UCF-101/ucfTrainTestlist/classInd.txt", sep=' ', header=None, engine="pyarrow")
    label_data.columns=['index', 'labels']
    label_data = label_data.drop(['index'], axis=1)
    label_data.head()
    path=[]
    for label in label_data.labels.values:
        path.append('../data/images_ucf101/'+label+"/")

    print('\n Plotting the data...\n')
    plot_results.plot_ucf101(label_data)

    print('\n Loading images...\n')
    with np.load('./results/all_images.npz') as images_file:
        images = images_file['arr_0']

    print('\n Loading labels...\n')
    labels_list = load_label(path)

    #Train Test Split
    print('\n Splitting the data into training and test...\n')
    x_train, x_test, y_train, y_test=train_test_split(images, labels_list, test_size=0.06,
                                                    random_state=10)

    train_gen = our_utils.DataGenerator(x_train, utils.to_categorical(y_train), batch_size=32)
    test_gen = our_utils.DataGenerator(x_test, utils.to_categorical(y_test), batch_size=32)


    return train_gen, test_gen, label_data