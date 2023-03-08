"""
This moduel contains all the plots produced from the package.
"""
import argparse
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from e2evideo import utils
import random
import cv2

device = utils.get_device()

def plot_cae_training(data, network, color_channels):
    """
    Plot the CAE training results, it results in plotting the original Vs. reconstruced image.
    """
    counter = 0
    for visual_images in tqdm(data):
        #  sending test images to device
        visual_images = visual_images.to(device)
        with torch.no_grad():
            #  reconstructing test images
            reconstructed_imgs = network(visual_images)
            #  sending reconstructed and images to cpu to allow for visualization
            reconstructed_imgs = reconstructed_imgs.cpu()
            visual_images = visual_images.cpu()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Original/Reconstructed')
        if color_channels == 1:
            ax2.imshow(reconstructed_imgs.reshape(224, 224, color_channels), cmap='gray')
        else:
            ax2.imshow(reconstructed_imgs.reshape(224, 224, color_channels))
        ax1.imshow(visual_images.squeeze())
        for ax_ in [ax1, ax2]:
            ax_.axis('off')
        file_name = './results/cae_' + str(counter) + '.jpg'
        counter += 1
        plt.show()
        plt.savefig(file_name)

def plot_ucf101(label_data):
    # Create a Matplotlib figure
    plt.figure(figsize = (30, 30))
    
    # Get Names of all classes in UCF101
    all_classes_names = label_data.labels.values
    
    # Generate a random sample of images each time the cell runs
    random_range = random.sample(range(len(all_classes_names[0:10])), 8)
    
    # Iterating through all the random samples
    for counter, random_index in enumerate(random_range, 1):
    
        # Getting Class Name using Random Index
        selected_class_Name = all_classes_names[random_index]
    
        # Getting a list of all the video files present in a Class Directory
        video_files_names_list = os.listdir(f'./data/UCF-101/{selected_class_Name}')
    
        # Randomly selecting a video file
        selected_video_file_name = random.choice(video_files_names_list)
    
        # Reading the Video File Using the Video Capture
        video_reader = cv2.VideoCapture(f'./data/UCF-101/{selected_class_Name}/{selected_video_file_name}')
        
        # Reading The First Frame of the Video File
        _, bgr_frame = video_reader.read()
    
        # Closing the VideoCapture object and releasing all resources. 
        video_reader.release()
    
        # Converting the BGR Frame to RGB Frame 
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
        # Adding The Class Name Text on top of the Video Frame.
    
        cv2.rectangle(rgb_frame, (30, 200), (290, 240), (255,255,255), -1)
        cv2.putText(rgb_frame, selected_class_Name, (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (160,32,240), 2, cv2.LINE_AA)
        cv2.rectangle(rgb_frame, (1, 1), (320, 240), (160,32,240), 10)

        # Assigning the Frame to a specific position of a subplot
        plt.subplot(5, 4, counter)
        plt.imshow(rgb_frame)
        plt.axis('off')
        # save image to a file 
        plt.savefig('./results/ucf101.jpg')
        print('image saved to file')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_frame')
    parser.add_argument('--nn')
    parser.add_argument('--color_channels')
    args = parser.parse_args()
    plot_cae_training(args.data_frame, args.nn, args.color_channels)
