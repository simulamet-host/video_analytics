"""
Convolutional autoencoder module for unsupervised feature extraction.
Part of the code adapted from https://blog.paperspace.com/convolutional-autoencoder/
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from e2evideo import plot_results
from e2evideo import our_utils

device = our_utils.get_device()
CCH = 3
# The parameter 'latent dim' refers to the size of the bottleneck = 1000
#  defining encoder
class Encoder(nn.Module):
    """
    Encoder class.
    """
    def __init__(self, in_channels=CCH, out_channels=64, latent_dim=1000, act_fn=nn.ReLU(), img_width=224, img_height=224):
        super().__init__()

        self.net = nn.Sequential(
            # Conv-1
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # (224, 244, 64)
            nn.BatchNorm2d(out_channels),
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (112, 112, 64)
            # Conv-2
            nn.Conv2d(out_channels, 2 * out_channels, 3, padding=1), # (112, 112, 128)
            nn.BatchNorm2d(2 * out_channels),
            act_fn,
            nn.Conv2d(2* out_channels, 2 * out_channels, 3, padding=1),
            nn.BatchNorm2d(2 * out_channels),
            act_fn,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (56, 56, 128)
            #Conv-3
            nn.Conv2d(2 * out_channels, 4*out_channels, 3, padding=1), # (56, 56, 256)
            nn.BatchNorm2d(4*out_channels),
            act_fn,
            nn.Conv2d(4 * out_channels, 4*out_channels, 3, padding=1),
            nn.BatchNorm2d(4*out_channels),
            act_fn,
            nn.Conv2d(4 * out_channels, 4*out_channels, 3, padding=1),
            nn.BatchNorm2d(4*out_channels),
            act_fn,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (28, 28 , 256)
            # Conv-4
            nn.Conv2d(4*out_channels, 8*out_channels, 3, padding=1), # (28, 28 , 512)
            nn.BatchNorm2d(8*out_channels),
            act_fn,
            nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
            nn.BatchNorm2d(8*out_channels),
            act_fn,
            nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
            nn.BatchNorm2d(8*out_channels),
            act_fn,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (14, 14 , 512)
            #Conv-5
            nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1), # (14, 14 , 512)
            nn.BatchNorm2d(8*out_channels),
            act_fn,
            nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
            nn.BatchNorm2d(8*out_channels),
            act_fn,
            nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
            nn.BatchNorm2d(8*out_channels),
            act_fn,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # (7, 7 , 512)
            nn.Flatten(),
            nn.Linear(8*out_channels*7*7, latent_dim), #(1000)
            act_fn)
    def forward(self, in_):
        """
        Forward function in the encoder.
        """
        in_ = in_.view(-1, CCH, img_width, img_height)
        output = self.net(in_)
        return output

#  defining decoder
class Decoder(nn.Module):
    """
    Decoder class.
    """
    def __init__(self, in_channels=CCH, out_channels=64, latent_dim=1000, act_fn=nn.ReLU()):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 8*out_channels*7*7),
            act_fn)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1), # (7, 7, 512)
            act_fn,
            nn.ConvTranspose2d(8*out_channels, 4*out_channels, 3, padding=1,
                            stride=2, output_padding=1), # (14, 14, 256)
            act_fn,
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1,
                            stride=2, output_padding=1), # (28, 28, 128)
            act_fn,
            nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, stride=2,
                output_padding=1), # (56, 56, 64)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1,
            stride=2, output_padding=1), # (112, 112, 64)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1,
                            stride=2, output_padding=1), # (224, 224, 64)
            act_fn,
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1), # (224, 224, 3)
            nn.Sigmoid()
            )
    def forward(self, bottleneck):
        """
        Forward function in the decoder.
        """
        output = self.linear(bottleneck)
        output = output.view(-1, 8*self.out_channels, 7, 7)
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
            print(images)
            #  zeroing gradients
            self.optimizer.zero_grad()
            #  sending images to device
            images = images.to(device)
            #  reconstructing images
            output = self.network(images)
            #  computing loss
            loss = training_args['loss_function'](output, images.view(-1, CCH, img_width, img_height))
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
                test_loss = training_args['loss_function'](output, test_images.view(-1,
                                            CCH, img_width, img_height))
            # LOGGING
            log_dict['test_loss_per_batch'].append(test_loss.item())
        print(f'training_loss: {round(loss.item(), 4)} test_loss: {round(test_loss.item(), 4)}')
        plot_results.plot_cae_training(loaders['visual_loader'], self.network, CCH)
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
