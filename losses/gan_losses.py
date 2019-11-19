import torch
import torch.nn as nn
import torch.nn.functional as F


def vanilla_gan_loss_G(d_fake):
    """Vanilla GAN loss for the generator network"""
    return F.binary_cross_entropy(
        input=d_fake,
        target=torch.ones_like(d_fake))

def vanilla_gan_loss_D_real(d_real):
    """Vanilla GAN loss for the discriminator network (real data)"""
    return F.binary_cross_entropy(
        input=d_real,
        target=torch.ones_like(d_real))

def vanilla_gan_loss_D_fake(d_fake):
    """Vanilla GAN loss for the discriminator network (fake/synthesized data)"""
    return F.binary_cross_entropy(
        input=d_fake,
        target=torch.zeros_like(d_fake))
