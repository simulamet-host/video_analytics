"""
This file contains the utility functions for the project. It contains the following functions:
1. get_device: This function check if the machine has GPU, then it returns a cuda device object.
    Either CPU or GPU.
"""
from torch.cuda import is_available
from torch import device # pylint: disable= E0611

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
