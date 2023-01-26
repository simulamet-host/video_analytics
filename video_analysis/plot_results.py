"""
This moduel contains all the plots produced from the package.
"""
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils import get_device

device = get_device()

def plot_CAE_training(df, network, color_channels):
    """
    Plot the CAE training results, it results in plotting the original Vs. reconstruced image. 
    """
    counter = 0
    for visual_images in tqdm(df):
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
        ax1.imshow(visual_images.squeeze())
        ax2.imshow(reconstructed_imgs.reshape(224, 224, color_channels))
        for ax_ in [ax1, ax2]:
            ax_.axis('off')
        file_name = './results/cae_' + str(counter) + '.jpg' 
        
        counter += 1
        plt.show()
        plt.savefig(file_name)
