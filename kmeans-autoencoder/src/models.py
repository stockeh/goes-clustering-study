import torch
import torch.nn as nn
import numpy as np

from typing import List
from torch import tensor as Tensor
from torch.nn import functional as F

class LinearAutoencoder(nn.Module):
    """Linear Autoencoder"""

    def __init__(self, in_shape: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        """
        Args:
        :in_shape: input shape
        :latent_dim: dim of latent space.
        :hidden_dims: units in conv layers, e.g., [512, 256, 128, 10]
        """
        super(LinearAutoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        hidden_dims.append(latent_dim)

        in_features = in_shape

        #### Build Encoder ####
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=h_dim),
                    nn.Tanh())
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)

        #### Build Decoder ####
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            h_dim = hidden_dims[i + 1]
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=h_dim),
                    nn.Tanh())
            )
            in_features = h_dim

        modules.append(
            nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_shape),
                nn.Sigmoid())
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, X):
        Z = self.encoder(X)
        Y = self.decoder(Z)
        return Y


class ColvolutionalAutoencoder(nn.Module):
    """Colvolutional Autoencoder"""

    def __init__(self, in_shape: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 ker_str_pad: List = None,
                 **kwargs) -> None:
        """
        Args:
        :in_shape: shape of input and output tensor, e.g., [C x H x W]
        :latent_dim: dim of latent space.
        :hidden_dims: units in conv layers, e.g., [4, 8, 16]
        :ker_str_pad: size of (kernels, stride, padding)
                      e.g., [(10, 3, 1), (8, 2, 0), (3, 3, 0)]
        """
        super(ColvolutionalAutoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [4, 8, 16]
        if ker_str_pad is None:
            ker_str_pad = [(10, 3, 1), (8, 2, 0), (3, 3, 0)]

        ker_str_pad = ker_str_pad
        self.hidden_dims = hidden_dims

        self.output_size_previous = in_shape[1]
        in_channels = in_shape[0]

        #### Build Encoder ####
        modules = []
        for (h_dim, ksp) in zip(self.hidden_dims, ker_str_pad):
            kernel, stride, padding = ksp
            modules.append(
                nn.Sequential(
                    # nn.Conv2d(...)
                    # ((n + 2p - k) / s) + 1
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=kernel, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            self.output_size_previous = (self.output_size_previous + 2 * padding - kernel) // stride + 1
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(hidden_dims[-1] * self.output_size_previous ** 2, latent_dim),
                nn.BatchNorm1d(num_features=latent_dim))
        )

        self.encoder = nn.Sequential(*modules)

        #### Build Decoder ####

        self.decoder_input = nn.Sequential(
                                nn.Linear(latent_dim, self.hidden_dims[-1] * self.output_size_previous ** 2))

        self.hidden_dims.reverse()
        ker_str_pad.reverse()

        modules = []
        for i in range(len(self.hidden_dims) - 1):
            kernel, stride, padding = ker_str_pad[i]
            modules.append(
                nn.Sequential(
                    # nn.ConvTranspose2d(...)
                    # (n - 1)s - 2p + (k - 1) + op + 1
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=kernel,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=padding),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[-1],
                                   self.hidden_dims[-1],
                                   kernel_size=ker_str_pad[-1][0],
                                   stride=ker_str_pad[-1][1],
                                   padding=ker_str_pad[-1][2]),
                nn.BatchNorm2d(self.hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(self.hidden_dims[-1], out_channels=in_shape[0],
                          kernel_size=3, stride=1, padding=1),
                nn.Sigmoid())
        )

        self.decoder = nn.Sequential(*modules)

        self.hidden_dims.reverse()
        ker_str_pad.reverse()

    def encode(self, X: Tensor) -> List[Tensor]:
        """
        Encodes X into latent vector
        :param input: Input tensor to encoder [N x C x H x W]
        :return: Latent vector
        """
        return self.encoder(X)


    def decode(self, Z: Tensor) -> Tensor:
        """
        Maps the given latent vector onto the image space.
        :param Z: [B x D]
        :return: [B x C x H x W]
        """
        Z = self.decoder_input(Z)
        Z = Z.view(-1, self.hidden_dims[-1], self.output_size_previous, self.output_size_previous)
        return self.decoder(Z)

    def forward(self, X):
        Z = self.encode(X)
        Y = self.decode(Z)
        return Y


class VariationalAutoencoder(torch.nn.Module):
    """Unfortunately the VAE does not work particularly well on data
    where n_samples << dimensionality.
    Modified from:
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    https://github.com/pytorch/examples/blob/master/vae/main.py
    """
    def __init__(self,
                 in_shape: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 ker_str_pad: List = None,
                 **kwargs) -> None:
        """
        :in_shape: shape of input and output tensor, e.g., [C x H x W]
        :latent_dim: dim of latent space.
        :hidden_dims: units in conv layers, e.g., [4, 8, 16]
        :ker_str_pad: size of (kernels, stride, padding)
                      e.g., [(10, 3, 1), (8, 2, 0), (3, 3, 0)]
        """
        super(VariationalAutoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        if ker_str_pad is None:
            ker_str_pad = [(10, 3, 1), (8, 2, 0), (3, 3, 0)]

        ker_str_pad = ker_str_pad
        self.hidden_dims = hidden_dims

        self.output_size_previous = in_shape[1]
        in_channels = in_shape[0]

        #### Build Encoder ####
        modules = []
        for (h_dim, ksp) in zip(self.hidden_dims, ker_str_pad):
            kernel, stride, padding = ksp
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=kernel, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            self.output_size_previous = (self.output_size_previous + 2 * padding - kernel) // stride + 1
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(self.hidden_dims[-1] * self.output_size_previous ** 2, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * self.output_size_previous ** 2, latent_dim)

        #### Build Decoder ####
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * self.output_size_previous ** 2)

        self.hidden_dims.reverse()
        ker_str_pad.reverse()

        modules = []
        for i in range(len(self.hidden_dims) - 1):
            kernel, stride, padding = ker_str_pad[i]
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=kernel,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=padding),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=ker_str_pad[-1][0],
                                               stride=ker_str_pad[-1][1],
                                               padding=ker_str_pad[-1][2]),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], out_channels=in_shape[0],
                                      kernel_size=3, stride=1, padding=1),
                            nn.Sigmoid())

        self.hidden_dims.reverse()
        ker_str_pad.reverse()

    def encode(self, X: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        Z = self.encoder(X)
        Z = torch.flatten(Z, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(Z)
        log_var = self.fc_var(Z)

        return [mu, log_var]

    def decode(self, Z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param Z: [B x D]
        :return: [B x C x H x W]
        """
        Y = self.decoder_input(Z)
        Y = Y.view(-1, self.hidden_dims[-1], self.output_size_previous, self.output_size_previous)
        Y = self.decoder(Y)
        Y = self.final_layer(Y)
        return Y

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: Mean of the latent Gaussian [B x D]
        :param logvar: Standard deviation of the latent Gaussian [B x D]
        :return: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        Y = args[0]
        X = args[1]
        mu = args[2]
        log_var = args[3]

        MSE = F.mse_loss(Y, X)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

        return MSE + KLD

    def forward(self, X: Tensor) -> List[Tensor]:
        """
        Computes the forward pass through the network
        :param X: input sample [B x C x H x W]
        :return: Y, X, mu, log_var
        """
        mu, log_var = self.encode(X)
        Z = self.reparameterize(mu, log_var)
        return [self.decode(Z), X, mu, log_var]


def get_model(X_shape, args):
    if 'cnn' in args.model:
        return ColvolutionalAutoencoder(X_shape, args.latent_dim,
                                        args.hidden_dims, args.ker_str_pad)
    if 'vae' in args.model:
        return VariationalAutoencoder(X_shape, args.latent_dim,
                                      args.hidden_dims, args.ker_str_pad)
    elif 'lin' in args.model:
        return LinearAutoencoder(np.product(X_shape), args.latent_dim,
                                 args.hidden_dims)
    else:
        print(f'WARN: {args.model} is not supported.')
