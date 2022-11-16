"""
PyTorch GAN Zoo - Fully Connected Networks
Author: Vahid Mirjalili
"""
import numpy as np
import torch.nn as nn

from ganzoo.misc import torch_ops


#===================================#
# FC-small models                   #
#===================================#

class FCSmall_Generator(nn.Module):
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

    def forward(self, z):
        return self.net(z)


class FCSmall_Discriminator(nn.Module):
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
        nn.Linear(num_hidden_units, num_hidden_units),
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
        nn.Linear(num_hidden_units, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, 1)]

    if activation == 'sigmoid':
        # use sigmoid only for vanilla gan
        layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)

    return model

#===================================#
# FC-Skip-Connection models         #
#===================================#
class FCSkipConnect_Generator(nn.Module):
    def __init__(
            self,
            num_z_units: int,
            num_hidden_units: int,
            output_image_dim: int,
            output_image_channels: int,
            p_drop: float):

        super().__init__()

        output_image_size = np.prod(
            [output_image_dim, output_image_dim, output_image_channels])
        self.l1 = nn.Linear(num_z_units, num_hidden_units)
        self.l2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.l3 = nn.Linear(num_hidden_units, output_image_size)
        self.drop = nn.Dropout(p_drop)
        self.reshape = torch_ops.ReshapeNonBatch(
            (output_image_channels, output_image_dim, output_image_dim))
        self.activ = nn.Tanh()

    def forward(self, z):
        h1 = nn.functional.relu(self.l1(z))
        h2 = nn.functional.relu(self.l2(h1))
        dpo = self.drop(h2)
        out = self.activ(self.l3(dpo + h1))
        return self.reshape(out)


class FCSkipConnect_Discriminator(nn.Module):
    def __init__(
            self,
            input_feature_dim: int,
            num_hidden_units: int,
            p_drop: float,
            activation: str):

        super().__init__()

        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_feature_dim, num_hidden_units)
        self.l2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.l3 = nn.Linear(num_hidden_units, 1)
        self.drop = nn.Dropout(p_drop)
        if activation == 'sigmoid':
            self.activ = nn.Sigmoid()

    def forward(self, x):
        xflat = self.flatten(x)
        h1 = nn.functional.relu(self.l1(xflat))
        h2 = nn.functional.relu(self.l2(h1))
        dpo = self.drop(h2)
        logits = self.l3(dpo + h1)
        return self.activ(logits)

#===================================#
# FC-Large models                   #
#===================================#
class FCLarge_Generator(nn.Module):
    def __init__(
            self,
            num_z_units: int,
            num_hidden_units: int,
            output_image_dim: int,
            output_image_channels: int,
            p_drop: float):

        super().__init__()

        def _fc_block(
                in_feat: int,
                out_feat: int,
                normalize: bool = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        img_shape = [
            output_image_channels, output_image_dim, output_image_dim
        ]
        self.net = nn.Sequential(
            *_fc_block(num_z_units, num_hidden_units, normalize=False),
            *_fc_block(num_hidden_units, num_hidden_units * 2),
            *_fc_block(num_hidden_units * 2, num_hidden_units * 4),
            *_fc_block(num_hidden_units * 4, num_hidden_units * 8),
            nn.Linear(num_hidden_units * 8, int(np.prod(img_shape))),
            nn.Tanh(),
            torch_ops.ReshapeNonBatch(img_shape)
        )

    def forward(self, z):
        return self.net(z)


class FCLarge_Discriminator(nn.Module):
    def __init__(
            self,
            input_feature_dim: int,
            num_hidden_units: int,
            p_drop: float,
            activation: str):

        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(input_feature_dim, num_hidden_units),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_hidden_units, num_hidden_units // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_hidden_units // 2, 1)
        ]
        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#...
