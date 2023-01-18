"""
This moduel contains all the plots produced from the package.
"""
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils import get_device

device = get_device()

def plot_CAE_training(df, network):
    """
    Plot the CAE training results, it results in plotting the original Vs. reconstruced image. 
    """
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
        ax2.imshow(reconstructed_imgs.reshape(32, 32, 3))
        for ax_ in [ax1, ax2]:
            ax_.axis('off')
        plt.show()
