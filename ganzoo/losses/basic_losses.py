"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""

import torch
import torch.nn.functional as F


def vanilla_gan_lossfn_G(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Basic GAN loss for the generator network
    """
    return F.binary_cross_entropy(
        input=d_fake,
        target=torch.ones_like(d_fake))


def vanilla_gan_lossfn_D_real(d_real: torch.Tensor) -> torch.Tensor:
    """
    Basic GAN loss for the discriminator network
    (real data)
    """
    return F.binary_cross_entropy(
        input=d_real,
        target=torch.ones_like(d_real))


def vanilla_gan_lossfn_D_fake(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Basic GAN loss for the discriminator network
    (fake/synthesized data)
    """
    return F.binary_cross_entropy(
        input=d_fake,
        target=torch.zeros_like(d_fake))
