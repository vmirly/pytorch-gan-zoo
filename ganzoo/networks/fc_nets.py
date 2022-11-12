"""
PyTorch GAN Zoo - Fully Connected Networks
Author: Vahid Mirjalili
"""

import torch.nn as nn

from ganzoo.misc import torch_ops


class FC_Generator(nn.Module):
    def __init__(
            self,
            num_z_units: int,
            num_hidden_units: int,
            output_image_dim: int,
            output_image_channels: int,
            p_drop: float):

        super().__init__()

        self.net = make_fully_connected_generator(
            num_z_units=num_z_units,
            num_hidden_units=num_hidden_units,
            output_image_dim=output_image_dim,
            output_image_channels=output_image_channels,
            p_drop=p_drop)

    def forward(self, x):
        return self.net(x)


class FC_Discriminator(nn.Module):
    def __init__(
            self,
            input_feature_dim: int,
            num_hidden_units: int,
            p_drop: float,
            activation: str):

        super().__init__()

        self.net = make_fully_connected_discriminator(
            input_feature_dim=input_feature_dim,
            num_hidden_units=num_hidden_units,
            p_drop=p_drop,
            activation=activation)

    def forward(self, x): 
        return self.net(x)


def make_fully_connected_generator(
        num_z_units: int,
        num_hidden_units: int,
        output_image_dim: int,
        output_image_channels: int,
        p_drop: float) -> nn.Module:

    """
    Paramaters
    ==
    num_z_units: int
      Size of latent dimension

    num_hidden_units: int
      Number of hidden units for the first FC layer

    output_image_dim: int
      Size (width and height) of the output image

    output_image_channels: int
      Number of channels on the output image

    p_drop: float
      Dropout probability

    Returns
    ==
    model: torch.nn.Module
      Fully connected generator network
    """

    output_image_size = (
        output_image_channels * output_image_dim * output_image_dim)

    model = nn.Sequential(
        nn.Linear(num_z_units, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, output_image_size),
        nn.Tanh(),
        torch_ops.ReshapeNonBatch(
            (output_image_channels,
             output_image_dim,
             output_image_dim)))

    return model


def make_fully_connected_discriminator(
        input_feature_dim: int,
        num_hidden_units: int,
        p_drop: float,
        activation: str) -> nn.Module:
    """
    Paramaters
    ==
    input_feature_dim: int
      Number of input features (width * height * channels)

    num_hidden_units: int
      Number of hidden units for the first FC layer

    p_drop: float
      Dropout probability

    activation: str choices=['sigmoid', 'none']

    Returns
    ==
    model: torch.nn.Module
      Fully connected discriminator network
    """

    layers = [
        nn.Flatten(),
        nn.Linear(input_feature_dim, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, 1)]

    if activation == 'sigmoid':
        # use sigmoid only for vanilla gan
        layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)

    return model
