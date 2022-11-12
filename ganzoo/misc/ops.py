"""
PyTorch GAN Zoo - Misc. Operations
Author: Vahid Mirjalili
"""

import random
from collections import deque
from typing import Callable, Deque, Union

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR


def get_latent_sampler(
        z_dim: int,
        z_distribution: str,
        make_4d: bool) -> Callable[[int], torch.Tensor]:

    def sample_z_uniform(batch_size: int) -> torch.Tensor:
        z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, z_dim))
        z = torch.from_numpy(z).float()
        if make_4d:  # 4-dimensional tensor
            return z.unsqueeze(2).unsqueeze(3)
        return z

    def sample_z_normal(batch_size: int) -> torch.Tensor:
        z = np.random.randn(batch_size, z_dim)
        z = torch.from_numpy(z).float()
        if make_4d:  # 4-dimensional tensor
            return z.unsqueeze(2).unsqueeze(3)
        return z

    if z_distribution == 'uniform':
        return sample_z_uniform
    # normal distribution
    return sample_z_normal


def unnormalize(images: np.ndarray) -> np.ndarray:
    n_channels = images.shape[1]
    mean = [0.5] * n_channels
    scale = [0.5] * n_channels
    mean = np.array(mean)[:, np.newaxis, np.newaxis]
    scale = np.array(scale)[:, np.newaxis, np.newaxis]
    return images * scale + mean


def recover_image_rgb(
        images: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray) -> np.ndarray:
    """
    Function to unnormalize a batch of images, and clip
    pixel intensties to the range [0-255].

    Input images must be a 4D NumPy array, and the channel dimension must
    be the last dimension.
    """

    mean_extended = mean.reshape(1, 1, 1, 3)
    std_extended = std.reshape(1, 1, 1, 3)
    unnorm_images = images * std_extended + mean_extended

    return (unnorm_images * 255).clip(0, 255).astype('uint8')


def get_scheduler(
        optimizer: torch.optim.Optimizer,
        lr_scheduler_type: str,
        step_size: Union[int, None],
        start_decay_step: Union[int, None],
        end_decay_step: Union[int, None]
        ) -> Union[StepLR, LambdaLR]:

    def linear_lr(step):
        n_decay_steps = end_decay_step - start_decay_step
        factor = (step - start_decay_step) / (n_decay_steps + 1.0)
        return 1.0 - max(0, factor)

    if lr_scheduler_type == 'step':
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=step_size,  # type: ignore
            gamma=0.1)

    else:
        lambda_lr = linear_lr

        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda_lr)  # type: ignore

    return scheduler


class ReplayMemory():
    """
    A replay memory to store the generated examples
    to be re-feed to a discriminator network.

    *Note*: please detach the tensors before storing them,
    otherwise, the occupied memory expands quickly.
    """

    def __init__(
            self,
            max_size: int) -> None:

        self.buffer: Deque = deque(maxlen=max_size)

    def memorize(
            self,
            batch_examples: torch.Tensor) -> None:

        for example in batch_examples:
            self.buffer.append(example.unsqueeze(0).clone())

    def choices(
            self,
            num_samples: int) -> torch.Tensor:

        choices = random.choices(self.buffer, k=num_samples)

        return torch.cat(choices, dim=0)
