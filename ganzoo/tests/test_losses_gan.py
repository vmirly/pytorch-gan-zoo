"""
PyTorch GAN Zoo - unit-tests for basic-loss functions
Author: Vahid Mirjalili
"""

import torch
import numpy as np

from ganzoo.losses import basic_losses


def test_vanilla_gan_loss_G():
    t = torch.tensor([1.0, 0.5, 1.0])
    loss_g = basic_losses.vanilla_gan_lossfn_G(t)
    assert np.isclose(loss_g, 0.2310, atol=1e-4)


def test_vanilla_gan_loss_D_fake():
    t = torch.tensor([0.1, 0.5, 0.1])
    loss_d_fake = basic_losses.vanilla_gan_lossfn_D_fake(t)
    assert np.isclose(loss_d_fake, 0.3013, atol=1e-4)


def test_vanilla_gan_loss_D_real():
    t = torch.tensor([1.0, 0.5, 1.0])
    loss_d_real = basic_losses.vanilla_gan_lossfn_D_real(t)
    assert np.isclose(loss_d_real, 0.2310, atol=1e-4)
