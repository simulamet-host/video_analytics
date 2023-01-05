"""
Module for feature extraction from images.
some code blocks are partially adapted from https://blog.paperspace.com/convolutional-autoencoder/
"""
import argparse
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm_regular
#from image_preprocessing import get_images
import matplotlib.pyplot as plt
#import cv2

from conv_autoencoder import *

def extract_each_class(dataset):
    """
    This function searches for and returns
    one image per class
    """
    images = []
    ITERATE = True
    i = 0
    j = 0
    while ITERATE:
        for label in tqdm_regular(dataset.targets):
            if label==j:
                images.append(dataset.data[i])
                print(f'class {j} found')
                i+=1
                j+=1
                if j==10:
                    ITERATE = False
            else:
                i+=1
    return images

#  defining dataset class
class CustomCIFAR10(Dataset):
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
    if args.dataset_name == "CIFAR10":
        print('Extracting features from ',  args.dataset_name)
        training_set = Datasets.CIFAR10(root='../data/', download=True,
                                transform=transforms.ToTensor())
        
        validation_set = Datasets.CIFAR10(root='../data/', download=True, train=False,
                                transform=transforms.ToTensor())
    #  extracting training images
    training_images = [x for x in training_set.data]
    #  extracting validation images
    validation_images = [x for x in validation_set.data]
    #  extracting test images for visualization purposes
    test_images = extract_each_class(validation_set)

    #  training model
    model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))

    if args.dataset_name == "CIFAR10":
        #  creating pytorch datasets
        training_data = CustomCIFAR10(training_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        validation_data = CustomCIFAR10(validation_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_data = CustomCIFAR10(test_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    training_args = {'loss_function': nn.MSELoss(), 'epochs': 1 , 'batch_size': 64,
                  'training_set': training_data, 'validation_set': validation_data, 'test_set': test_data}
    log_dict = model.train(training_args) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='CIFAR10')

    args = parser.parse_args()
    main(args)

    #images_ = get_images('./images/', '*.jpg', True, (224, 224))
    # Show the first image in the directory
    #img = images_[0]
    #plt.imshow(img)
    #plt.show()