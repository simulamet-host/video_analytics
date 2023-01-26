"""
Module for feature extraction from images.
Some code blocks are partially adapted from https://blog.paperspace.com/convolutional-autoencoder/
"""
import argparse
import numpy as np
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from image_preprocessing import get_images
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

from conv_autoencoder import *

#  defining dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transforms!=None:
            image = self.transforms(image)
        return image

def main(args):
    print('Extracting features from ',  args.dataset_name)
    if args.dataset_name == "CIFAR10":        
        training_set = Datasets.CIFAR10(root='../data/', download=True,
                                transform=transforms.ToTensor())
        test_set = Datasets.CIFAR10(root='../data/', download=True, train=False,
                                transform=transforms.ToTensor())
        #  extracting training images
        training_images = [x for x in training_set.data]
        #  extracting test images
        test_images = [x for x in test_set.data]
        
        #  creating pytorch datasets
        training_data = CustomDataset(training_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_data = CustomDataset(test_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_data = CustomDataset(test_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    if args.dataset_name == "handwashing":
        _dir = "./images/ex_images/"
        _images = get_images(_dir, '*.jpg', True, (224, 224), False)    
        print('images size: ' , _images.size)
        training_data, test_data = train_test_split( _images, test_size=0.2, random_state=42)
        visual_data, test_data = train_test_split(test_data, test_size=0.98, random_state=42)
    #  training model
    model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))
    training_args = {'loss_function': nn.MSELoss(), 'epochs': 1000 , 'batch_size': 10,
                'training_set': training_data, 'test_set': test_data, 'visual_set': visual_data}
    log_dict = model.train(training_args) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='handwashing')
    args = parser.parse_args()
    main(args)
