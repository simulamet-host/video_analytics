"""
Object_detection module. It contains functions that are designed
to perform object detection tasks on specific datasets.
This initial version is tailored specifically for the object detection dataset.
"""
import argparse
import time
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import models, layers, callbacks
from e2evideo import plot_results, load_ucf101

def object_detection_model(train_dataset, test_dataset):
    """
    This function is used to perform object detection on the dataset.
    """
    print("Object Detection")
    # Define the model
    x_train, _ = next(train_dataset)

    convlstm_model = models.Sequential()
    convlstm_model.add(layers.BatchNormalization(momentum=0.8, input_shape=(x_train.shape[1],
                x_train.shape[2], x_train.shape[3], 3)))
    convlstm_model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                         data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    convlstm_model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same',
                                           data_format='channels_last'))
    convlstm_model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    convlstm_model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                         data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    convlstm_model.add(layers.BatchNormalization(momentum=0.8))
    convlstm_model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same',
                                           data_format='channels_last'))
    convlstm_model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    convlstm_model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                        data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    convlstm_model.add(layers.BatchNormalization(momentum=0.8))
    convlstm_model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same',
                                           data_format='channels_last'))
    convlstm_model.add(layers.TimeDistributed(layers.Dropout(0.3)))
    convlstm_model.add(layers.ConvLSTM2D(filters = 16, kernel_size=(3,3), activation='LeakyReLU',
                        data_format='channels_last', return_sequences=True, recurrent_dropout=0.2))
    convlstm_model.add(layers.BatchNormalization(momentum=0.8))
    convlstm_model.add(layers.MaxPooling3D(pool_size=(1,2,2), padding='same',
                                           data_format='channels_last'))
    convlstm_model.add(layers.TimeDistributed(layers.Dropout(0.3)))
    convlstm_model.add(layers.Flatten())
    convlstm_model.add(layers.Dense(4096,activation="relu"))
    convlstm_model.add(layers.Dense(101, activation='softmax'))
    convlstm_model.summary()

    #compile model
    convlstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    #Model training
    early_stop= callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min',
                                restore_best_weights=True)
    history_ = convlstm_model.fit(train_dataset,  epochs=150,
                        validation_data=test_dataset, callbacks=[early_stop])
    # save the model
    convlstm_model.save('./results/models/convlstm_model.h5')
    return  history_

if __name__ == '__main__':
    print('Video Classification using ConvLSTM')
    # calculate and print the time needed to run the code below using time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices= ['train', 'test'],
                        help='train or test')
    args = parser.parse_args()
    
    #Train the model
    if args.mode == 'train':
        print('\n Loading data...\n')
        train_gen, test_gen, label_data = load_ucf101.load_ucf101()
        # save test_gen to a file
        print('\n Saving test_gen to a file...\n')
        np.save('./results/test_gen.npy', test_gen)
        print('\n Training the model...\n')
        HISTORY = object_detection_model(train_gen, test_gen)
        print('\n Plotting the accuracy and loss...\n')
        plot_results.plot_accuracy(HISTORY)

    #Test the model
    else:
        # load the test generator from the npy file
        print('\n Loading test_gen from a file...\n')
        test_gen = np.load('./results/test_gen.npy')
        x_test, y_test = next(test_gen)
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
