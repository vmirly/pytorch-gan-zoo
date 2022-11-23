"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""
import sys
import logging

import torch


def wgan_lossfn_G(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein GAN loss for the generator network
    """
    return -d_fake.mean().view(1)


def wgan_lossfn_D_real(d_real: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein GAN loss for the discriminator network
    (real data)
    """
    return -d_real.mean().view(1)


def wgan_lossfn_D_fake(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein GAN loss for the discriminator network
    (fake/synthesized data)
    """
    return d_fake.mean().view(1)


def wgan_gradient_penalty(
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: torch.nn.Module,
        device: torch.device,
        epsilon: float) -> torch.Tensor:

    if len(real) != len(fake):
        logging.error(
            'The size of real and fake tensors are not the same.\n'
            f'size of real tensor: {len(real)}, '
            f'size of fake tensor: {len(fake)}')
        sys.exit(1)

    batch_size = len(real)

    # calc. x_hat: random interpolation of real and fake

    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    x_hat = alpha * real + (1 - alpha) * fake.detach()
    x_hat.requires_grad = True

    # calc. d_hat: discriminator output on x_hat
    d_hat = discriminator(x_hat)

    # calc. gradients of d_hat vs. x_hat
    grad_values = torch.ones(d_hat.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=grad_values,
        create_graph=True,
        retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)

    # calc. norm of gradients, adding epsilon to prevent 0 values
    gradients_norm = torch.sqrt(
        torch.sum(gradients ** 2, dim=1) + epsilon)

    return ((gradients_norm - 1) ** 2).mean()


def wgan_lipschitz_penalty(
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: torch.nn.Module,
        device: torch.device,
        epsilon: float) -> torch.Tensor:
    """
    Lipschitz penalty for WGAN
    Paper: https://arxiv.org/pdf/1709.08894.pdf
    """

    if len(real) != len(fake):
        logging.error(
            'The size of real and fake tensors are not the same.\n'
            f'size of real tensor: {len(real)}, '
            f'size of fake tensor: {len(fake)}')
        sys.exit(1)

    batch_size = len(real)

    # calc. x_hat: random interpolation of real and fake

    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    x_hat = alpha * real + (1 - alpha) * fake.detach()
    x_hat.requires_grad = True

    # calc. d_hat: discriminator output on x_hat
    d_hat = discriminator(x_hat)

    # calc. gradients of d_hat vs. x_hat
    grad_values = torch.ones(d_hat.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=grad_values,
        create_graph=True,
        retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)

    # calc. norm of gradients, adding epsilon to prevent 0 values
    gradients_norm = torch.sqrt(
        torch.sum(gradients ** 2, dim=1) + epsilon)

    zero = torch.zeros(size=(len(gradients_norm), 1)).to(device)

    one_sided_gradients = torch.max(zero, (gradients_norm - 1))

    return (one_sided_gradients ** 2).mean()
