"""
PyTorch GAN Zoo - Basic FC GAN
Author: Vahid Mirjalili
"""

import os
import sys
import time
import logging
import argparse

import numpy as np
import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
import mlflow

from ganzoo.misc import ops
from ganzoo.misc import data_ops
from ganzoo.losses import basicgan_losses
from ganzoo.nn_modules import fc_nets
from ganzoo.constants import data_constants, network_constants


def main(args: argparse.Namespace) -> int:

    sample_z = ops.get_latent_sampler(
        z_dim=args.z_dim,
        z_distribution=args.z_distribution,
        make_4d=False)

    # setup dataset
    if args.dataset == 'existing_dataset':
        image_dim = args.image_dim
        image_channels = data_constants.IMAGE_CHANNELS[args.dataset_name]
        transform, msg = data_ops.get_transforms(
            initial_resize=None,
            center_crop_size=None,
            random_crop_size=None,
            proba_hflip=None,
            final_resize=image_dim,
            normalize_mean=[0.5] * image_channels,
            normalize_std=[0.5] * image_channels)
        if msg:
            logging.error(msg)
            return 1

        dataset = data_ops.get_dataset(
            dataset_name=args.dataset_name,
            transform=transform,
            mode='train')

    else:  # custom dataset
        image_dim = args.image_dim
        image_channels = args.image_channels
        transform, msg = data_ops.get_transforms(
            initial_resize=None,
            center_crop_size=args.center_crop_size,
            random_crop_size=args.random_crop_size,
            final_resize=image_dim,
            proba_hflip=network_constants.PROBA_HFLIP,
            normalize_mean=[0.5] * image_channels,
            normalize_std=[0.5] * image_channels)
        if msg:
            logging.error(msg)
            return 1

        dataset = data_ops.ImageFolder(
            image_dir=args.train_image_dir,
            labels_file_csv=None,
            transform=transform,
            extensions=(args.image_extensions,),
            include_subdirs=args.include_subdirs)

    num_iters_per_epoch = len(dataset) // args.batch_size  # type: ignore

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True)

    generator = fc_nets.make_fully_connected_generator(
        num_z_units=args.z_dim,
        num_hidden_units=args.num_hidden_units,
        output_image_dim=image_dim,
        output_image_channels=image_channels,
        p_drop=args.p_drop)

    input_size = image_dim * image_dim * image_channels
    discriminator = fc_nets.make_fully_connected_discriminator(
        input_feature_dim=input_size,
        num_hidden_units=args.num_hidden_units,
        p_drop=args.p_drop,
        activation=network_constants.DISC_ACTIVATIONS[args.gan_loss])

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    print(generator)
    print(discriminator)
    print(
        '============\nNumber of learnable parameters\n'
        f'Generator {utils.count_parameters(generator)} '
        f'Discriminator {utils.count_parameters(discriminator)}\n'
        '============')

    lr = network_constants.LEARNING_RATES['vanilla']
    beta1 = network_constants.ADAM_BETA1
    beta2 = network_constants.ADAM_BETA2['vanilla']
    optim_g = optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(beta1, beta2))
    optim_d = optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(beta1, beta2))

    fixed_z = sample_z(network_constants.NUM_EVAL_IMAGES).to(device)

    def _training_step_G(batch_z):
        generator.train()
        generator.zero_grad()

        gen_images = generator(batch_z)

        d_fake = discriminator(gen_images)
        loss_g = vanillagan_losses.vanilla_gan_lossfn_G(d_fake)

        loss_g.backward()
        optim_g.step()

        return loss_g.item()

    def _training_step_D(batch_z, batch_real_images):
        discriminator.train()
        discriminator.zero_grad()

        gen_images = generator(batch_z)

        d_real = discriminator(batch_real_images)
        loss_d_real = vanillagan_losses.vanilla_gan_lossfn_D_real(d_real)

        d_fake = discriminator(gen_images)
        loss_d_fake = vanillagan_losses.vanilla_gan_lossfn_D_fake(d_fake)

        loss_total = 0.5 * (loss_d_real + loss_d_fake)

        loss_total.backward()
        optim_d.step()

        return loss_total.item()

    step = 0
    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        list_gloss, list_dloss = [], []
        for idx, (batch_real, _) in enumerate(train_loader):
            batch_real = batch_real.to(device)

            batch_z = sample_z(len(batch_real)).to(device)

            loss_g = _training_step_G(batch_z)
            loss_d = _training_step_D(batch_z, batch_real)
            list_gloss.append(loss_g)
            list_dloss.append(loss_d)

            # Logging training losses
            if not idx % args.visualize_freq:
                step += 1
                avg_lg = np.mean(list_gloss)
                avg_ld = np.mean(list_dloss)

                if args.mlflow_tracking_enabled:
                    mlflow.log_metric('losses/loss_g', avg_lg, step=step)
                    mlflow.log_metric('losses/loss_d', avg_ld, step=step)

                generator.eval()
                gen_images = generator(fixed_z).detach().cpu()

                if args.dataset == 'mnist':
                    gen_images = 1 - ops.unnormalize(gen_images)
                else:
                    gen_images = ops.unnormalize(gen_images)

                grid_generated = torchvision.utils.make_grid(
                    gen_images, nrow=4)

                filename = os.path.join(
                    args.checkpoint_dir,
                    'frames',
                    f'frame_{step:05d}.jpg')
                utils.save_grid_image(grid_generated, filename)

                if args.mlflow_tracking_enabled:
                    mlflow.log_artifact(filename, artifact_path='generated-images')

                list_gloss, list_dloss = [], []

            if idx % args.log_freq == 0:
                log_msg = (
                    f'Epoch {epoch:3d}/{args.num_epochs} '
                    f'Iter {idx:6d}/{num_iters_per_epoch} '
                    f'| Losses  G: {loss_g:.4f}  D: {loss_d:.4f}')
                print(log_msg)

        print(f'Time Elapsed: {time.time() - start_time:.1f} sec')

        if epoch % args.save_freq == 0:
            state_dict = {
               'epoch': epoch,
               'model_state_dict': generator.state_dict()}
            output_file = os.path.join(
                args.checkpoint_dir,
                f'model-{epoch}.tch')
            torch.save(state_dict, output_file)

    if args.animate_progress:
        status = utils.create_animation_from_imagedir(
            image_dir=os.path.join(args.image_dir, 'frames'),
            image_extension='.jpg',
            output_filename=os.path.join(args.image_dir, 'animation.gif'))

        return status

    return 0


if __name__ == '__main__':
    args = parser.parse_train_args(sys.argv[1:])

    # mlflow
    if args.mlflow_tracking_enabled:
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=args.run_name)

    status = main(args)

    if args.mlflow_tracking_enabled:
        mlflow.end_run()

    sys.exit(status)
