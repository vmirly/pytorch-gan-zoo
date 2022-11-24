"""
PyTorch GAN Zoo - unit-tests for basic-loss functions
Author: Vahid Mirjalili
"""

import torch
import numpy as np

from ganzoo.losses import basic_losses
from ganzoo.losses import wgan_losses


bx = torch.randn((2, 4))
net = torch.nn.Linear(4, 1)

def test_vanilla_gan_loss_G():
    t = torch.tensor([1.0, 0.5, 1.0])
    loss_g = basic_losses.vanilla_gan_lossfn_G(t)
    assert np.isclose(loss_g, 0.2310, atol=1e-4)
    # test the backward
    out_d = torch.sigmoid(net(bx))
    loss_g = basic_losses.vanilla_gan_lossfn_G(out_d)
    loss_g.backward()


def test_vanilla_gan_loss_D_fake():
    t = torch.tensor([0.1, 0.5, 0.1])
    loss_d_fake = basic_losses.vanilla_gan_lossfn_D_fake(t)
    assert np.isclose(loss_d_fake, 0.3013, atol=1e-4)
    # test the backward
    out_d = torch.sigmoid(net(bx))
    loss_d_fake = basic_losses.vanilla_gan_lossfn_D_fake(out_d)
    loss_d_fake.backward()


def test_vanilla_gan_loss_D_real():
    t = torch.tensor([1.0, 0.5, 1.0])
    loss_d_real = basic_losses.vanilla_gan_lossfn_D_real(t)
    assert np.isclose(loss_d_real, 0.2310, atol=1e-4)
    # test the backward
    out_d = torch.sigmoid(net(bx))
    loss_d_real = basic_losses.vanilla_gan_lossfn_D_real(out_d)
    loss_d_real.backward()


def test_wgan_lossfn_G():
    t = torch.tensor([1.2, -0.5, 2.5, -2.2])
    loss_g = wgan_losses.wgan_lossfn_G(t)
    assert np.isclose(loss_g, -0.25, atol=1e-4)
    # test the backward
    out_d = torch.sigmoid(net(bx))
    loss_g = wgan_losses.wgan_lossfn_G(out_d)
    loss_g.backward()


def test_wgan_lossfn_D_real():
    t = torch.tensor([1.2, -0.5, 2.5, -2.2])
    loss_d_real = wgan_losses.wgan_lossfn_D_real(t)
    assert np.isclose(loss_d_real, -0.25, atol=1e-4)
    # test the backward
    out_d = net(bx)
    loss_d_real = wgan_losses.wgan_lossfn_D_real(out_d)
    loss_d_real.backward()


def test_wgan_lossfn_D_fake():
    t = torch.tensor([1.2, -0.5, 2.5, -2.2])
    loss_d_fake = wgan_losses.wgan_lossfn_D_fake(t)
    assert np.isclose(loss_d_fake, 0.25, atol=1e-4)
    # test the backward
    out_d = net(bx)
    loss_d_fake = wgan_losses.wgan_lossfn_D_fake(out_d)
    loss_d_fake.backward()


def test_wgan_lossfn_gradpenalty():
    rt = torch.tensor([[1.2, -0.5, 2.5, -2.2]])
    ft = torch.tensor([[0.6, -0.25, 1.25, -1.1]])
    discriminator = torch.nn.Linear(4, 1)

    loss_d_gp = wgan_losses.wgan_gradient_penalty(rt, ft, discriminator)

    # re-compute
    w = discriminator.weight
    w_norm = torch.sqrt(torch.sum(w ** 2, dim=1))
    recomp_gp = ((w_norm - 1) ** 2).mean()
    assert np.isclose(loss_d_gp.detach(), recomp_gp.detach(), atol=1e-4)

    # test the backward
    loss_d_gp.backward()


def test_wgan_lossfn_lipschitzpenalty():
    rt = torch.tensor([[1.2, -0.5, 2.5, -2.2]])
    ft = torch.tensor([[0.2, 0.5, 1.2, 1.1]])
    discriminator = torch.nn.Linear(4, 1)
    def const_init(model):
        for name, param in model.named_parameters():
            param.data.uniform_(-2.0, 2.0)
    const_init(discriminator)

    loss_d_lp = wgan_losses.wgan_lipschitz_penalty(rt, ft, discriminator)

    # re-compute
    w = discriminator.weight
    w_rss = torch.sqrt(torch.sum(w ** 2))  # toot-sum-squared
    zero = torch.zeros(1).type_as(rt)
    w_rss_onesided = torch.max(zero, (w_rss - 1))
    recomp_lp = (w_rss_onesided ** 2).mean()
    assert np.isclose(loss_d_lp.detach(), recomp_lp.detach(), atol=1e-4)

    # test the backward
    loss_d_lp.backward()
