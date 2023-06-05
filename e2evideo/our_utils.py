"""
This file contains the utility functions for the project. It contains the following functions:
1. get_device: This function check if the machine has GPU, then it returns a cuda device object.
    Either CPU or GPU.
"""
import numpy as np
from torch.cuda import is_available
from torch import device # pylint: disable= E0611
import tensorflow as tf

def print_nn_layers(net, input__):
    """
    This function print the nn layers.
    """
    for layer in net:
        print('\n\n')
        print('Layer: ', layer)
        input_ = layer(input__)
        print(input_clea.size())

def get_device(dev_id=0):
    """
    This function check if the machine has GPU, then it returns a cuda device object.
    Either CPU or GPU.
    """
    #  configuring device
    if is_available():
        # pylint: disable=E1101
        dev_name = 'cuda:' + str(dev_id)
        which_device = device(dev_name)
        print('Running on the GPU')
    else:
        which_device = device('cpu') # pylint:disable=E1101
        print('Running on the CPU')
    return which_device

class DataGenerator(tf.keras.utils.Sequence):
    """DataGenerator class"""
    def __init__(self, x_set, y_set, batch_size):
        self.x_, self.y_ = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x_[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)
