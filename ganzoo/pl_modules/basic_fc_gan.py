"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""

from typing import Callable
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_lightning as pl

from ganzoo.networks import fc_nets
from ganzoo.losses import basic_losses
from ganzoo.constants import network_constants


class PLBasicGANFC(pl.LightningModule):
    def __init__(
            self,
            num_z_units: int,
            num_hidden_units: int,
            image_dim: int,
            image_channels: int,
            p_drop: float,
            lr: float,
            beta1: float,
            beta2: float,
            z_sampler: Callable[[None], torch.Tensor]):

        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.z_sampler = z_sampler

        self.generator = fc_nets.make_fully_connected_generator(
            num_z_units=num_z_units,
            num_hidden_units=num_hidden_units,
            output_image_dim=image_dim,
            output_image_channels=image_channels,
            p_drop=p_drop)

        self.discriminator = fc_nets.make_fully_connected_discriminator(
            input_feature_dim=np.prod([image_dim, image_dim, image_channels]),
            num_hidden_units=num_hidden_units,
            p_drop=p_drop,
            activation=network_constants.DISC_ACTIVATIONS['vanilla'])

        self.criterion_G = basic_losses.vanilla_gan_lossfn_G
        self.criterion_D_real = basic_losses.vanilla_gan_lossfn_D_real
        self.criterion_D_fake = basic_losses.vanilla_gan_lossfn_D_fake

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        optim_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr, betas=(self.beta1, self.beta2))
        optim_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr, betas=(self.beta1, self.beta2))
        return [optim_G, optim_D], []

    def training_step(self, batch_data, batch_idx, optimizer_idx):
        def _training_step_G(batch_z):
            gen_images = self(batch_z)
            d_fake = self.discriminator(gen_images)
            loss_g = self.criterion_G(d_fake)
            self.log("G-loss", loss_g, prog_bar=True)
            return loss_g

        def _training_step_D(batch_z, batch_real_images):
            gen_images = self(batch_z)
            d_real = self.discriminator(batch_real_images)
            loss_d_real = self.criterion_D_real(d_real)
            d_fake = self.discriminator(gen_images)
            loss_d_fake = self.criterion_D_fake(d_fake)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            self.log("D-loss", loss_d, prog_bar=True)
            return loss_d

        batch_imgs, _ = batch_data
        batch_z = self.z_sampler(len(batch_imgs)).type_as(batch_imgs)

        if optimizer_idx == 0:  # train G
            loss_g = _training_step_G(batch_z)
            return {'loss': loss_g}
        else: # train D
            loss_d = _training_step_D(batch_z, batch_imgs)
            return {'loss': loss_d}

    def validation_step(self, batch_data, batch_idx):
        res_g = self.training_step(batch_data, batch_idx, 0)
        res_d = self.training_step(batch_data, batch_idx, 1)
        return {'loss_g': res_g['loss'], 'loss_d': res_d['loss']}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss_g = torch.tensor([
            x['loss_g'] for x in val_step_outputs]).mean()
        avg_val_loss_d = torch.tensor([
            x['loss_d'] for x in val_step_outputs]).mean()
        return {'val_loss_g': avg_val_loss_g, 'val_loss_d': avg_val_loss_d}

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=32)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=32)
        return val_loader

    def prepare_data(self):
        # prepare_data only works on one GPU
        # we download the data once
        datasets.MNIST(
            'data', train=True, download=True,
            transform=transforms.ToTensor()
        )

    def setup(self, stage):
        ds = datasets.MNIST(
            'data', train=True, download=True,
            transform=transforms.ToTensor()
        )
        self.train_data, self.val_data = random_split(ds, [55000, 5000])
