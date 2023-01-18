"""
Convolutional autoencoder module for unsupervised feature extraction.
Part of the code adapted from https://blog.paperspace.com/convolutional-autoencoder/
Last updated January 2023
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from plot_results import plot_CAE_training
from tqdm import tqdm
from utils import get_device

device = get_device()

# The parameter 'latent dim' refers to the size of the bottleneck = 1000
#  defining encoder
class Encoder(nn.Module):
    """
    Encoder class.
    """
    def __init__(self, in_channels=3, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (16, 16)
            nn.Conv2d(out_channels, 2*out_channels, 3, padding=1), 
            act_fn,
            nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (8, 8)
            nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(4*out_channels*8*8, latent_dim),
            act_fn)
    def forward(self, in_):
        """
        Forward function in the encoder.
        """
        in_ = in_.view(-1, 3, 32, 32)
        output = self.net(in_)
        return output

#  defining decoder
class Decoder(nn.Module):
    """
    Decoder class.
    """
    def __init__(self, in_channels=3, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4*out_channels*8*8),
            act_fn)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
            act_fn,
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1,
                            stride=2, output_padding=1), # (16, 16)
            act_fn,
            nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1,
                            stride=2, output_padding=1), # (32, 32)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1) )
    def forward(self, bottleneck):
        """
        Forward function in the decoder.
        """
        output = self.linear(bottleneck)
        output = output.view(-1, 4*self.out_channels, 8, 8)
        output = self.conv(output)
        return output

#  defining autoencoder
class Autoencoder(nn.Module):
    """
    Autoencoder class.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(device)
        self.decoder = decoder
        self.decoder.to(device)
    def forward(self, in_):
        """
        Forward function for the autoencoder class.
        """
        encoded = self.encoder(in_)
        decoded = self.decoder(encoded)
        return decoded

class ConvolutionalAutoencoder():
    """
    Convolutional Autoencoder class.
    """
    def __init__(self, autoencoder):
        self.network = autoencoder
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
    def train(self, training_args):
        """
        Train method used to train the model.
        """
        #  creating log
        log_dict = {
            'training_loss_per_batch': [],
            'test_loss_per_batch': [],
            'visualizations': []
        }
        #  defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
        #  initializing network weights
        self.network.apply(init_weights)
        #  creating dataloaders
        loaders = dict.fromkeys(['train_loader', 'test_loader'], None)
        loaders['train_loader'] = DataLoader(training_args['training_set'],
                                            training_args['batch_size'])
        loaders['test_loader'] = DataLoader(training_args['test_set'],
                                            training_args['batch_size'])
        loaders['visual_loader'] = DataLoader(training_args['visual_set'])

        #  setting convnet to training mode
        self.network.train()
        self.network.to(device)
        for epoch in range(training_args['epochs']):
            print(f'Epoch {epoch+1}/{training_args["epochs"]}')
        #  TRAINING
        print('training...')
        for images in tqdm(loaders['train_loader']):
            #  zeroing gradients
            self.optimizer.zero_grad()
            #  sending images to device
            images = images.to(device)
            #  reconstructing images
            output = self.network(images)
            #  computing loss
            loss = training_args['loss_function'](output, images.view(-1, 3, 32, 32))
            #  calculating gradients
            loss.backward()
            #  optimizing weights
            self.optimizer.step()
            # LOGGING
            log_dict['training_loss_per_batch'].append(loss.item())
        # Testing
        print('testing...')
        for test_images in tqdm(loaders['test_loader']):
            with torch.no_grad():
                #  sending test images to device
                test_images = test_images.to(device)
                #  reconstructing images
                output = self.network(test_images)
                #  computing test loss
                test_loss = training_args['loss_function'](output, test_images.view(-1, 3, 32, 32))
            # LOGGING
            log_dict['test_loss_per_batch'].append(test_loss.item())

        print(f'training_loss: {round(loss.item(), 4)} test_loss: {round(test_loss.item(), 4)}')
        plot_CAE_training(loaders['visual_loader'], self.network)
        return log_dict

    def autoencode(self, in_):
        """
        Autoencoder.
        """
        return self.network(in_)
    def encode(self, in_):
        """
        Encoder method, take input image and output the bottleneck.
        """
        encoder = self.network.encoder
        return encoder(in_)
    def decode(self, bottleneck):
        """
        Decoder method, input bottleneck and output the reconstructed image.
        """
        decoder = self.network.decoder
        return decoder(bottleneck)
