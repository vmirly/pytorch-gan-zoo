"""
PyTorch GAN Zoo - Misc. PyTorch Operatons
Author: Vahid Mirjalili
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from typing import Callable, Tuple, Union


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def onehot_encoding(
        num_classes: int,
        device: torch.device = None) -> Callable[[torch.Tensor], torch.Tensor]:
    y = torch.eye(num_classes)
    if device is not None:
        y = y.to(device)

    def encode_to_onehot(labels):
        return y[labels]

    return encode_to_onehot


class OnehotEncoder(nn.Module):

    def __init__(
            self,
            num_classes: int) -> None:

        super().__init__()

        # define as module Paramater so that
        # it can be transferred to torch.device
        # using m.to(device)
        self.onehot_map = nn.Parameter(
            data=torch.eye(num_classes),
            requires_grad=False)

    def forward(
            self,
            inputs: torch.Tensor) -> torch.Tensor:

        return self.onehot_map[inputs]


class ReshapeNonBatch(nn.Module):
    """
    A custom layer for reshaping feature maps
    """
    def __init__(self, new_shape: Tuple):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.view(inputs.size(0), *self.new_shape)


class PairedRandomHorizontalFlip(object):
    """
    A class for applying random horizontal flip to a pair of images

    To preserve the alignment of the two images, random flipping must be
     applied to both or none
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image_pair):
        r = np.random.uniform(0, 1.0, size=None)

        if r > (1.0 - self.prob):
            return F.hflip(image_pair[0]), F.hflip(image_pair[1])
        else:
            return image_pair


class PairedRandomCrop(object):
    """
    A class for applying random crop to a pair of images

    To preserve the alignment of the two images, cropping coordinates
     must be the same for both images.

     - crop_size: Tuple[int]
     the crop size for height and width
    """

    def __init__(self, crop_size: Tuple[int, int]) -> None:
        self.crop_size = crop_size

    def __call__(
            self,
            image_pair: Tuple[Image.Image, Image.Image]
            ) -> Tuple[Image.Image, Image.Image]:
        width, height = image_pair[0].size
        # the two images must have the same size

        left = np.random.randint(0, width - self.crop_size[1], size=None)
        top = np.random.randint(0, height - self.crop_size[0], size=None)
        right = left + self.crop_size[1]
        bottom = top + self.crop_size[0]

        img1_crop = image_pair[0].crop((left, top, right, bottom))
        img2_crop = image_pair[1].crop((left, top, right, bottom))

        return img1_crop, img2_crop


class MaxoutFC(nn.Module):
    """
    Maxout activations for FC networks
    """
    def __init__(
            self,
            n_pieces: int,
            in_features: int,
            out_features: int,
            bias: bool) -> None:

        super().__init__()

        self.n_pieces = n_pieces

        self.main = []
        for _ in range(self.n_pieces):
            self.main.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias))

    def forward(
            self,
            inputs: torch.Tensor) -> torch.Tensor:

        max_output = self.main[0](inputs)
        for fc in self.main[1:]:
            max_output = torch.max(max_output, fc(inputs))

        return max_output


class MaxoutConv(nn.Module):
    """
    Maxout activation for CNNs
    """
    def __init__(
            self,
            n_pieces: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int]],
            bias: bool) -> None:

        super().__init__()

        self.n_pieces = n_pieces

        self.main = []
        for _ in range(self.n_pieces):
            self.main.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias))

    def forward(
            self,
            input_tensor: torch.Tensor) -> torch.Tensor:

        max_output = self.main[0](input_tensor)
        for conv in self.main[1:]:
            max_output = torch.max(max_output, conv(input_tensor))

        return max_output
