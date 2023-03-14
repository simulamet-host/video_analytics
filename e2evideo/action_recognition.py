"""
Object_detection module. It contains functions that are designed
to perform object detection tasks on specific datasets.
This initial version is tailored specifically for the object detection dataset.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras import models, layers, utils, callbacks

from plot_results import plot_ucf101, plot_accuracy

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def object_detection_model(train_dataset, test_dataset):
    """
    This function is used to perform object detection on the dataset.
    """
    print("Object Detection")
    # Define the model
    model = models.Sequential()
    model.add(layers.BatchNormalization(momentum=0.8, input_shape=(x_train.shape[1],
                x_train.shape[2], x_train.shape[3], 3)))
    model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                         data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                         data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                        data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.3)))
    model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                        data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation="relu"))
    model.add(layers.Dense(1, activation='softmax'))
    model.summary()

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    #Model training
    early_stop= callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min',
                                restore_best_weights=True)

    #print(x_train.shape, x_test.shape, np.array(y_train).shape, np.array(y_test).shape)
    #y_train = to_categorical([y_train])
    #y_train = y_train.reshape(y_train.shape[1], y_train.shape[0])
    #y_train = np.array(tf.stack(y_train))
    #print(y_train)
    #print(y_train.shape)

    history = model.fit(train_dataset , batch_size=32, epochs=5,
                        validation_data=(test_dataset), callbacks=[early_stop])

    # save the model
    model.save('./results/models/convlstm_model.h5')
    return  history

if __name__ == '__main__':
    print('Video Classification using ConvLSTM')
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # calculate and print the time needed to run the code below using time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices= ['train', 'test'],
                        help='train or test')
    args = parser.parse_args()

    print('\n Loading data...\n')
    label_data = pd.read_csv("../data/UCF-101/ucfTrainTestlist/classInd.txt", sep=' ', header=None, engine="pyarrow")
    label_data.columns=['index', 'labels']
    label_data = label_data.drop(['index'], axis=1)
    label_data.head()
    path=[]
    for label in label_data.labels.values:
        path.append('../data/images_ucf101/'+label+"/")

    print('\n Plotting the data...\n')
    plot_ucf101(label_data)

    # load images from file in the same folder
    print('\n Loading images...\n')
    images_file = np.load('./results/all_images.npz')
    images = images_file['arr_0']
    print(images.shape)
    print('\n Loading labels...\n')
    labels_list = load_label(path)

    #Train Test Split
    print('\n Splitting the data into training and test...\n')
    x_train, x_test, y_train, y_test=train_test_split(images, labels_list, test_size=0.06,
                                                      random_state=10)

    # implement tensorflow input data pipeline
    print('\n Implementing tensorflow input data pipeline...\n')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, utils.to_categorical(y_train)))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, utils.to_categorical(y_test)))

    #Train the model
    if args.mode == 'train':
        print('\n Training the model...\n')
        history = object_detection_model(train_dataset, test_dataset)
        print('\n Plotting the accuracy and loss...\n')
        plot_accuracy(history)
    else:
        # load model from file
        print('\n Loading the model...\n')
        model = tf.keras.models.load_model('./results/models/convlstm_model.h5')
        # evaluate the model
        print('\n Evaluating the model...\n')
        y_pred = model.predict(x_test)
        predicted_classes=[]
        for i in range(len(y_test)):
            predicted_classes.append(np.argmax(y_pred[i]))
        print(accuracy_score(y_test, predicted_classes))

    end_time = time.time()
    print("Time taken to run the code: ", end_time - start_time)

    #plot_confusion_matrix(y_test, predicted_classes)
