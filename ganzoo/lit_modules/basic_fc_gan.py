"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""

from typing import Callable, Tuple
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_lightning as pl

from ganzoo.nn_modules import fc_nets
from ganzoo.losses import basic_losses, wgan_losses
from ganzoo.constants import defaults
from ganzoo.misc import ops


def get_fc_networks(
        num_z_units: int,
        num_hidden_units: int,
        image_dim: int,
        image_channels: int,
        p_drop: float,
        network_type: str,
        loss_type: str) -> Tuple[torch.nn.Module]:

    activation = defaults.DISC_ACTIVATIONS[loss_type]
    if network_type == 'fc-small':
        generator = fc_nets.FCSmall_Generator(
            num_z_units=num_z_units,
            num_hidden_units=num_hidden_units,
            output_image_dim=image_dim,
            output_image_channels=image_channels,
            p_drop=p_drop)

        discriminator = fc_nets.FCSmall_Discriminator(
            input_feature_dim=np.prod([image_dim, image_dim, image_channels]),
            num_hidden_units=num_hidden_units,
            p_drop=p_drop,
            activation=activation)

    elif network_type == 'fc-skip':
        generator = fc_nets.FCSkipConnect_Generator(
            num_z_units=num_z_units,
            num_hidden_units=num_hidden_units,
            output_image_dim=image_dim,
            output_image_channels=image_channels,
            p_drop=p_drop)

        discriminator = fc_nets.FCSkipConnect_Discriminator(
            input_feature_dim=np.prod([image_dim, image_dim, image_channels]),
            num_hidden_units=num_hidden_units,
            p_drop=p_drop,
            activation=activation)

    elif network_type == 'fc-large':
        generator = fc_nets.FCLarge_Generator(
            num_z_units=num_z_units,
            num_hidden_units=num_hidden_units,
            output_image_dim=image_dim,
            output_image_channels=image_channels,
            p_drop=p_drop)

        discriminator = fc_nets.FCLarge_Discriminator(
            input_feature_dim=np.prod([image_dim, image_dim, image_channels]),
            num_hidden_units=num_hidden_units,
            p_drop=p_drop,
            activation=activation)

    return generator, discriminator


class LitBasicGANFC(pl.LightningModule):
    def __init__(
            self,
            num_z_units: int,
            z_distribution: str,
            num_hidden_units: int,
            image_dim: int,
            image_channels: int,
            p_drop: float,
            lr: float,
            beta1: float,
            beta2: float,
            network_type: str,
            loss_type: str):

        super().__init__()
        self.save_hyperparameters()

        self.z_sampler = ops.get_latent_sampler(
            z_dim=num_z_units,
            z_distribution=z_distribution,
            make_4d=False)

        self.fixed_z = self.z_sampler(batch_size=defaults.VALIDATION_SIZE)

        self.generator, self.discriminator = get_fc_networks(
            num_z_units=num_z_units,
            num_hidden_units=num_hidden_units,
            image_dim=image_dim,
            image_channels=image_channels,
            p_drop=p_drop,
            network_type=network_type,
            loss_type=loss_type)

        if loss_type == 'vanilla':
            self.criterion_G = basic_losses.vanilla_gan_lossfn_G
            self.criterion_D_real = basic_losses.vanilla_gan_lossfn_D_real
            self.criterion_D_fake = basic_losses.vanilla_gan_lossfn_D_fake
        elif loss_type in ('wgan', 'wgan-gp', 'wgan-lp'):
            self.criterion_G = wgan_losses.wgan_lossfn_G
            self.criterion_D_real = wgan_losses.wgan_lossfn_D_real
            self.criterion_D_fake = wgan_losses.wgan_lossfn_D_fake
            if loss_type == 'wgan-gp':
                self.criterion_regularization = wgan_losses.wgan_gradient_penalty
            else:
                self.criterion_regularization = wgan_losses.wgan_lipschitz_penalty
        else:
            pass  # TODO

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        optim_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        optim_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
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
            #if self.hparams.loss_type == 'basic':
            loss_d = loss_d_real + loss_d_fake
            #loss_d_reg = None
            #elif self.hparams.loss_type in ('wgan', 'wgan-gp', 'wgan-lp'):
            #    loss_d = loss_d_real + loss_d_fake
            #        loss_d_reg = None
            self.log("D-loss", loss_d, prog_bar=True)

            if self.hparams.loss_type in ('wgan-gp', 'wgan-lp'):
                loss_d_reg = self.criterion_regularization(
                    batch_real_images, gen_images, self.discriminator)
                # within torch.no_grad context, the output is None
                if loss_d_reg is not None:
                    self.log("D-loss_reg", loss_d_reg, prog_bar=True)
                    return loss_d + loss_d_reg
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

    def on_validation_epoch_end(self):
        batch_z = self.fixed_z.type_as(next(self.generator.parameters()))
        val_gen_imgs = self(batch_z)
        val_gen_imgs = ops.unnormalize_torch(val_gen_imgs)
        grid = torchvision.utils.make_grid(
            val_gen_imgs, normalize=True, valuerange=(-1, 1))
        self.logger.experiment.add_image(
            "generated_images", grid, self.current_epoch)
