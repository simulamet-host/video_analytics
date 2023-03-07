"""
Object_detection module. It contains functions that are designed
to perform object detection tasks on specific datasets.
This initial version is tailored specifically for the object detection dataset.
"""
import argparse
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from e2evideo import image_preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_label(datasets):
    """
    """
    label_index=0
    labels=[]
    #Iterate through each foler corresponding to category
    for folder in datasets:
        for file in tqdm(os.listdir(folder)):
            labels.append(label_index)
        label_index+=1
    return np.array(labels, dtype='int8')

def object_detection_model(x_train, y_train, x_test, y_test, frames_per_video):
    """
    This function is used to perform object detection on the dataset.
    """
    print("Object Detection")
    # Define the model
    model = Sequential()
    model.add(BatchNormalization(momentum=0.8, input_shape=(x_train.shape[1],x_train.shape[2],
                                                            x_train.shape[3], 3)))
    model.add(ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                         data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                         data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                        data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                        data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(Flatten())
    model.add(Dense(4096,activation="relu"))
    model.add(Dense(1, activation='softmax'))
    model.summary()

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    #Model training
    es = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    print(x_train.shape, x_test.shape, np.array(y_train).shape, np.array(y_test).shape)
    #y_train = to_categorical([y_train])
    #y_train = y_train.reshape(y_train.shape[1], y_train.shape[0])
    #y_train = np.array(tf.stack(y_train))
    #print(y_train)
    #print(y_train.shape)
    
    history = model.fit(x_train,  to_categorical(y_train) , batch_size=32, epochs=5,
                        validation_data=(x_test, to_categorical(y_test)), callbacks=[es])

    # save the model
    model.save('./results/models/convlstm_model.h5')
    return  history

def plot_accuracy(history):
    """
    This function is used to plot the accuracy of the model.
    """
    #Plot the graph to check training and testing accuracy over the period of time
    plt.figure(figsize=(13,5))
    plt.title("Accuracy vs Epochs")
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='best')
    plt.savefig('./results/accuracy_vs_epochs.png')
    plt.show()

def plot_frames_and_predictions(label_data):
    """
    This function is used to plot the frames and predictions of the model.
    """
    plt.figure(figsize = (30, 30))
    # Get Names of all classes in UCF101
    all_classes_names = label_data.labels.values
    # Generate a random sample of images each time the cell runs
    random_range = random.sample(range(len(all_classes_names[0:10])), 5)
    # Iterating through all the random samples
    for counter, random_index in enumerate(random_range, 1):
        # Getting Class Name using Random Index
        selected_class_Name = all_classes_names[random_index]
        # Getting a list of all the video files present in a Class Directory
        video_files_names_list = os.listdir(f'../data/UCF-101/{selected_class_Name}')
        # Randomly selecting a video file
        selected_video_file_name = random.choice(video_files_names_list)
        # Reading the Video File Using the Video Capture
        video_reader = cv2.VideoCapture(f'../data/UCF-101/{selected_class_Name}/{selected_video_file_name}')
        # Reading The First Frame of the Video File
        _, bgr_frame = video_reader.read()
        # Closing the VideoCapture object and releasing all resources.
        video_reader.release()
        # Converting the BGR Frame to RGB Frame
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # Adding The Class Name Text on top of the Video Frame.
        cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Assigning the Frame to a specific position of a subplot
        plt.subplot(5, 4, counter)
        plt.imshow(rgb_frame)
        plt.axis('off')
    plt.savefig('./results/UCF101.png')

def plot_confusion_matrix(y_test, predicted_classes):
    """
    This function is used to plot the confusion matrix of the model.
    """
    #Confusion Matrix
    plt.figure(figsize=(25,25))
    plt.title("Confusion matrix")
    cm=confusion_matrix(y_test, predicted_classes)
    plt.imshow(cm)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center")
    plt.savefig('./results/confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    print('Video Classification using ConvLSTM')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    label_data = pd.read_csv("../data/UCF-101/ucfTrainTestlist/classInd.txt", sep=' ', header=None)
    label_data.columns=['index', 'labels']
    label_data = label_data.drop(['index'], axis=1)
    label_data.head()

    path=[]
    for label in label_data.labels.values:
        path.append('../data/images_ucf101/'+label+"/")
    
    # load images from file in the same folder
    images = np.load('./results/all_images.npy')
    labels = load_label(path[:1])

    #Train Test Split
    x_train, x_test, y_train, y_test=train_test_split(images, labels, test_size=0.06, random_state=10)
        
    no_video = x_train.shape[0]
    history = object_detection_model(x_train, y_train, x_test, y_test, no_video)
    plot_accuracy(history)

    # load model from file
    model = tf.keras.models.load_model('./results/models/convlstm_model.h5')
  
    y_pred = model.predict(x_test)
    predicted_classes=[]
    for i in range(len(y_test)):
        predicted_classes.append(np.argmax(y_pred[i]))
    print(accuracy_score(y_test, predicted_classes))
        
    #TODO I need to re-write this function
    #plot_frames_and_predictions(label_data)
    
    #plot_confusion_matrix(y_test, predicted_classes)
