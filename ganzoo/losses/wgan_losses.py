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
        discriminator: torch.nn.Module) -> torch.Tensor:
        # epsilon: float) -> torch.Tensor:

    if len(real) != len(fake):
        logging.error(
            'The size of real and fake tensors are not the same.\n'
            f'size of real tensor: {len(real)}, '
            f'size of fake tensor: {len(fake)}')
        sys.exit(1)

    # calc. x_hat: random interpolation of real and fake
    alpha = torch.rand(real.size(0), 1, 1, 1).type_as(real)

    x_hat = alpha * real + (1 - alpha) * fake.detach()
    x_hat.requires_grad = True

    # calc. d_hat: discriminator output on x_hat
    d_hat = discriminator(x_hat)
    if not d_hat.grad_fn:
        return None

    # calc. gradients of d_hat vs. x_hat
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(d_hat.size()).type_as(real),
        create_graph=True,
        retain_graph=True)[0]
    grads = grads.view(x_hat.size(0), -1)

    # calc. norm of gradients
    grads_norm = grads.norm(p=2, dim=1)
    # calc. norm of gradients, adding epsilon to prevent 0 values
    # gradients_norm = torch.sqrt(
    #    torch.sum(gradients ** 2, dim=1) + epsilon)

    return ((grads_norm - 1) ** 2).mean()


def wgan_lipschitz_penalty(
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: torch.nn.Module) -> torch.Tensor:
        # epsilon: float) -> torch.Tensor:
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

    # calc. x_hat: random interpolation of real and fake
    alpha = torch.rand(real.size(0), 1, 1, 1).type_as(real)

    x_hat = alpha * real + (1 - alpha) * fake.detach()
    x_hat.requires_grad = True

    # calc. d_hat: discriminator output on x_hat
    d_hat = discriminator(x_hat)
    if not d_hat.grad_fn:
        return None

    # calc. gradients of d_hat vs. x_hat
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(d_hat.size()).type_as(real),
        create_graph=True,
        retain_graph=True)[0]
    grads = grads.view(x_hat.size(0), -1)

    # calc. norm of gradients
    grads_norm = grads.norm(p=2, dim=1)

    zero = torch.zeros(grads_norm.size(0), 1).type_as(grads_norm)
    one_sided_penalty = (torch.max(zero, (grads_norm - 1)) ** 2).mean()

    return one_sided_penalty
