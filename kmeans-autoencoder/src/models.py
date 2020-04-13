import torch
import torch.nn as nn
import numpy as np

class AutoencoderLinear(nn.Module):
    """Linear Autoencoder"""

    def __init__(self, X_shape, latent_dim):
        """
        Args:
            X_shape (int): shape of input and output vector.
            latent_dim (int): dimension of latent vector.
        """
        super(AutoencoderLinear, self).__init__()

        self.X_shape = X_shape
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.X_shape, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=self.latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=self.X_shape),
            nn.Sigmoid()
        )

    def forward(self, X):
        Z = self.encoder(X)
        Y = self.decoder(Z)
        return Y


class Autoencoder(nn.Module):
    """WARN: NOT FINISHED IMPLMENTING"""

    def __init__(self, n_channels):
        super(Autoencoder, self).__init__()
        # nn.Conv2d(
        #   n_units_previous, n_units,
        #   kernel_size, kernel_stride,
        #   padding)

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

        self.encoder = nn.Sequential()

        n_units_previous = self.n_channels
        output_size_previous = self.image_size
        n_layers = 0

        for (n_units, kernel_size_and_stride) in zip(self.n_units_in_conv_layers, self.kernels_size_and_stride):
                n_units_previous, output_size_previous = self._add_conv2d_tanh(n_layers,
                                        n_units_previous, output_size_previous, n_units,
                                        kernel_size_and_stride)

                n_layers += 1

        def _add_conv2d_tanh(self, n_layers, n_units_previous, output_size_previous,
                   n_units, kernel_size_and_stride):
            if len(kernel_size_and_stride) == 2:
                kernel_size, kernel_stride = kernel_size_and_stride
                padding = 0
            else:
                kernel_size, kernel_stride, padding = kernel_size_and_stride
            self.nnet.add_module(f'conv_{n_layers}', torch.nn.Conv2d(n_units_previous, n_units, kernel_size,
                                                                     kernel_stride, padding=padding))
            self.nnet.add_module(f'output_{n_layers}', torch.nn.ReLU())
            output_size_previous = (output_size_previous + 2 * padding - kernel_size) // kernel_stride + 1
            n_units_previous = n_units
            return n_units_previous, output_size_previous

    def forward(self, X):
        Z = self.encoder(X)
        Y = self.decoder(Z)
        return Y
