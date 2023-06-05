"""
Module for feature extraction from images.
Some code blocks are adapted from https://blog.paperspace.com/convolutional-autoencoder/
"""
import argparse
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import conv_autoencoder
import our_utils
import plot_results
from load_ucf11 import get_data
device = our_utils.get_device(dev_id=0)

def main(args_):
    """
    Main function. It is used to extract features from images.
    """
    # Read labels from file frames_labels.txt
    train_dataset, test_dataset = get_data(args_.labels_file, args_.images_array)
    if args_.mode == 'train':
        #  training model
        model = conv_autoencoder.ConvolutionalAutoencoder(conv_autoencoder.Autoencoder(
                conv_autoencoder.Encoder(), conv_autoencoder.Decoder()))
        training_args = {'loss_function': nn.BCELoss(), 'epochs': args_.epochs ,
                    'batch_size': args_.batch_size,
                    'training_set': train_dataset, 'test_set': test_dataset,
                    'visual_set': test_dataset}
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
