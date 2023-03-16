"""
Module for feature extraction from images.
Some code blocks are adapted from https://blog.paperspace.com/convolutional-autoencoder/
"""
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
from sklearn.model_selection import train_test_split
from e2evideo import image_preprocessing, conv_autoencoder, load_ucf101

#  defining dataset class
class CustomDataset(Dataset):
    """
    Custom dataset class. It is used to create a pytorch dataset from a list of images.
    """
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        return image

def main(args_):
    """
    Main function. It is used to extract features from images.
    """
    print('Extracting features from ',  args_.dataset_name)
    if args_.dataset_name == "CIFAR10":
        training_set = Datasets.CIFAR10(root='../data/', download=True,
                                transform=Transforms.ToTensor())
        test_set = Datasets.CIFAR10(root='../data/', download=True, train=False,
                                transform=Transforms.ToTensor())
        #  extracting training images
        training_images = list(training_set.data)
        #  extracting test images
        test_images = list(test_set.data)
        #  creating pytorch datasets
        training_data = CustomDataset(training_images,
                        transforms=Transforms.Compose([Transforms.ToTensor(),
                                    Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_data = CustomDataset(test_images, transforms=Transforms.Compose([Transforms.ToTensor(),
                                        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_data = CustomDataset(test_images, transforms=Transforms.Compose([Transforms.ToTensor(),
                                        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        visual_data = test_data
    if args_.dataset_name == "handwashing":
        _dir = "./images/"
        _images = image_preprocessing.get_images(_dir, '*.jpg', True, (224, 224), True)
        print('images size: ' , _images.size)
        training_data, test_data = train_test_split( _images, test_size=0.2, random_state=42)
        visual_data, test_data = train_test_split(test_data, test_size=0.98, random_state=42)
    
    if args_.dataset_name == "action_recognition":
        train_gen, test_gen, _ = load_ucf101.load_ucf101()
        training_data = train_gen
        test_data = test_gen
        visual_data = test_data

    #  training model
    model = conv_autoencoder.ConvolutionalAutoencoder(conv_autoencoder.Autoencoder(
            conv_autoencoder.Encoder(), conv_autoencoder.Decoder()))
    training_args = {'loss_function': nn.BCELoss(), 'epochs': 10 , 'batch_size': 16,
                'training_set': training_data, 'test_set': test_data, 'visual_set': visual_data}
    log_dict = model.train(training_args)
    print(log_dict)
    # save the model
    torch.save(model, '../results/encoder_model.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='handwashing')
    args_input = parser.parse_args()
    main(args_input)
