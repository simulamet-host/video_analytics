"""
This file contains the utility functions for the project. It contains the following functions:
1. get_device: This function check if the machine has GPU, then it returns a cuda device object.
    Either CPU or GPU.
"""
import numpy as np
from torch.cuda import is_available
from torch import device # pylint: disable= E0611
import tensorflow as tf

def print_NN_layers(net, input):
    for layer in net:
        print('\n\n')
        print('Layer: ', layer)
        input = layer(input)
        print(input.size())

def get_device():
    """
    This function check if the machine has GPU, then it returns a cuda device object.
    Either CPU or GPU.
    """
    #  configuring device
    if is_available():
        # pylint: disable=E1101
        which_device = device('cuda:0')
        print('Running on the GPU')
    else:
        which_device = device('cpu') # pylint:disable=E1101
        print('Running on the CPU')
    return which_device

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

