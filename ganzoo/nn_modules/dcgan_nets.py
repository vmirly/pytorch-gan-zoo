"""
PyTorch GAN Zoo
--

Convolutional generator and discriminator based on DCGAN architecture

Author: Vahid Mirjalili
"""

import torch
import torch.nn as nn


class DCGAN_Generator(nn.Module):
    """
    Generator network for 32x32 images

    Architecture adopted from DCGAN
    """

    def __init__(
            self,
            num_z_units: int,
            num_conv_filters: int,
            output_image_dim: int,
            output_image_channels: int):

        super().__init__()

        nf0 = num_z_units
        nch = output_image_channels
        nf0 = num_z_units
        nf1 = num_conv_filters * 4
        nf2 = num_conv_filters * 2
        nf3 = num_conv_filters

        # first block: project z
        z_projector = nn.Sequential(
            nn.ConvTranspose2d(nf0, nf1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=nf1),
            nn.ReLU(inplace=True))

        self.net = nn.Sequential(
            z_projector,
            ConvBlockUpsample(nf1, nf2, use_normalizer=True),
            ConvBlockUpsample(nf2, nf3, use_normalizer=True),
            nn.ConvTranspose2d(nf3, nch, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

# In WGAN discriminator (or also called critic),
# use of BatchNorm is discouraged, instead layer-norm is
# recommended by https://arxiv.org/pdf/1704.00028.pdf
# Here, we use InstanceNorm
class DCGAN_Discriminator(nn.Module):
    """
    Generator network for 32x32 or 64x64 images

    Architecture adopted from DCGAN
    """

    def __init__(
            self,
            image_dim: int,
            image_channels: int,
            num_conv_filters: int,
            activation: str):

        super().__init__()

        nf0 = image_channels
        nf1 = num_conv_filters
        nf2 = num_conv_filters * 2
        nf3 = num_conv_filters * 4

        if activation == 'sigmoid':
            last_activation = nn.Sigmoid()  # type: ignore
        else:  # linear activation
            last_activation = nn.Identity()  # type: ignore

        self.net = nn.Sequential(
            ConvBlockDownsample(nf0, nf1, False),
            ConvBlockDownsample(nf1, nf2, True),
            ConvBlockDownsample(nf2, nf3, True),
            nn.Conv2d(nf3, 1, 4, 1, 0, bias=False),
            last_activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze()

#=========================================#
#   helper function: ConvBlockUpsample    #
#=========================================#

class ConvBlockUpsample(nn.Module):
    """
    Convolution block for upsampling for use in Generator
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_normalizer: bool):

        super().__init__()

        conv_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
            bias=False)

        if use_normalizer:
            self.main = nn.Sequential(
                conv_layer,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else:
            self.main = nn.Sequential(
                conv_layer,
                nn.ReLU(inplace=True))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.main(inputs)


class ConvBlockDownsample(nn.Module):
    """
    Convolution block for downsampling inputs
    for use in Discriminator
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_normalizer: bool):

        super().__init__()

        # define convolution-layer:
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
            bias=False)

        # normalization layer
        if use_normalizer:
            self.main = nn.Sequential(
                conv_layer,
                nn.InstanceNorm2d(num_features=out_channels),
                nn.LeakyReLU(inplace=True, negative_slope=0.2))
        else:
            self.main = nn.Sequential(
                conv_layer,
                nn.LeakyReLU(inplace=True, negative_slope=0.2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.main(inputs)
