"""
Module for feature extraction from images.
Some code blocks are adapted from https://blog.paperspace.com/convolutional-autoencoder/
"""
import argparse
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
from sklearn.model_selection import train_test_split
from e2evideo import image_preprocessing
from e2evideo import conv_autoencoder
from e2evideo import load_ucf101
from e2evideo import our_utils
from e2evideo import plot_results
from e2evideo import load_ucf11

device = our_utils.get_device()

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

def cifar10_helper():
    """Helper function for CIFAR10 dataset."""
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
    return training_data, test_data, visual_data

def main(args_):
    """
    Main function. It is used to extract features from images.
    """
    print('Extracting features from ',  args_.dataset_name)
    if args_.dataset_name == "CIFAR10":
        _, test_data, visual_data = cifar10_helper()
    elif args_.dataset_name == "handwashing":
        _dir = "./images/"
        opt_dict = {'dir': _dir, 'img_format': '*.jpg', 'resize': True, 'img_width': 224,
                    'img_height': 224, 'gray_scale': True}
        opt_ = argparse.Namespace(**opt_dict)
        _images, _ = image_preprocessing.get_images(opt_)
        print('images size: ' , _images.size)
        _, test_data = train_test_split( _images, test_size=0.2, random_state=42)
        visual_data, test_data = train_test_split(test_data, test_size=0.98, random_state=42)
    elif args_.dataset_name == "action_recognition":
        x_train, x_test, _, _, _ = load_ucf101.load_ucf101(args_.data_folder, args_.images_array,
                                                           args_.no_classes)
        # pylint: disable=E1101
        x_train, x_test = torch.tensor(x_train).to(device), torch.tensor(x_test).to(device)

        #training_data = np.concatenate(train_gen[:, 0])
        #test_data = np.concatenate(test_gen[:, 0])
        #visual_data = test_data
    else:
        train_dataset, test_dataset = load_ucf11.get_data(args_.labels_file, args_.images_array)
    if args_.mode == 'train':
        #  training model
        model = conv_autoencoder.ConvolutionalAutoencoder(conv_autoencoder.Autoencoder(
                conv_autoencoder.Encoder(), conv_autoencoder.Decoder()))
        training_args = {'loss_function': nn.BCELoss(), 'epochs': args_.epochs,
                         'batch_size': args_.batch_size, 'training_set': train_dataset,
                         'test_set': test_dataset, 'visual_set': test_dataset}
        log_dict = model.train(training_args)
        print(log_dict)
    else:
        visual_data = DataLoader(test_dataset)
        #  loading model
        saved_model = torch.load('./checkpoints/model-11.pt')
        plot_results.plot_cae_training(visual_data, saved_model)

if __name__ == '__main__':
    # print the date and time of now
    print('Date and time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='action_recognition')
    parser.add_argument('--images_array', type=str, default='./results/ucf10.npz')
    parser.add_argument('--data_folder', type=str, default='../data/images_ucf10/')
    parser.add_argument('--labels_file', type=str, default=None)
    parser.add_argument('--no_classes', type=int, default=10)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    args_input = parser.parse_args()

    device = our_utils.get_device()
    main(args_input)
