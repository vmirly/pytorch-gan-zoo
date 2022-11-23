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

        nch = output_image_channels
        nf0 = num_z_units

        if output_image_dim == 32:
            nf1 = num_conv_filters * 4
            nf2 = num_conv_filters * 2
            nf3 = num_conv_filters
            nf4 = None
        else:  # 64x64
            nf1 = num_conv_filters * 8
            nf2 = num_conv_filters * 4
            nf3 = num_conv_filters * 2
            nf4 = num_conv_filters

        up_blocks = [
            UpsampleBlock(nf1, nf2, use_normalizer=True),
            UpsampleBlock(nf2, nf3, use_normalizer=True)
        ]
        if nf4:
            up_blocks.append(
                UpsampleBlock(nf3, nf4, use_normalizer=True))

        # first block: project z
        z_projector = nn.Sequential(
            nn.ConvTranspose2d(nf0, nf1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=nf1),
            nn.ReLU(inplace=True))

        self.net = nn.Sequential(
            z_projector,
            *up_blocks, # upsampling blocks
            nn.ConvTranspose2d(
                nf4 if nf4 else nf3, nch, 4, 2, 1, bias=False),
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
        if image_dim == 32:
            nf4 = None
        else:
            nf4 = num_conv_filters * 8

        if activation == 'sigmoid':
            last_activation = nn.Sigmoid()  # type: ignore
        else:  # linear activation
            last_activation = nn.Identity()  # type: ignore

        dw_blocks = [
            DownsampleBlock(nf0, nf1, False),
            DownsampleBlock(nf1, nf2, True),
            DownsampleBlock(nf2, nf3, True)
        ]
        if nf4:
            dw_blocks.append(DownsampleBlock(nf3, nf4, True))

        self.net = nn.Sequential(
            *dw_blocks,
            nn.Conv2d(nf4 if nf4 else nf3, 1, 4, 1, 0, bias=False),
            last_activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze()

#=========================================#
#   helper function: UpsampleBlock    #
#=========================================#

class UpsampleBlock(nn.Module):
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


class DownsampleBlock(nn.Module):
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
