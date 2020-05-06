import os
import pathlib
import time
import sys
import argparse

import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ops import constants
from networks import basic_gan_networks as nets
from utils import mnist_loader


def main(args):
    print(args)
    device = torch.device(
        "cuda:1" if (torch.cuda.is_available() and args.ngpu > 0)
        else "cpu")
    print('Device:', device)

    lossfn_G, lossfn_D_real, lossfn_D_fake = constants.GAN_LOSS[args.gan_loss]

    if args.fully_connected:
        generator = nets.make_generator(
            num_z_units=args.z_dim,
            num_hidden_units=args.hidden_dim,
            output_image_dim=args.image_dim,
            output_image_channels=args.image_channels,
            p_drop=args.p_drop).to(device)

        discriminator = nets.make_discriminator(
            input_feature_dim=args.image_dim * args.image_dim,
            num_hidden_units=args.hidden_dim,
            p_drop=args.p_drop).to(device)
    else:
        generator = nets.make_conv_generator(
            num_z_units=args.z_dim,
            num_filters=args.nf_generator,
            output_image_dim=args.image_dim,
            output_image_channels=args.image_channels).to(device)

        discriminator = nets.make_conv_discriminator(
            image_dim=args.image_dim,
            num_inp_channels=args.image_channels,
            num_filters=args.nf_discriminator).to(device)

    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.9))

    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.9))

    def training_step_G(batch_z):
        generator.train()
        generator.zero_grad()

        gen_images = generator(batch_z)

        d_fake = discriminator(gen_images)
        err_G = lossfn_G(d_fake)

        err_G.backward()
        optimizer_G.step()

        return {'errG': err_G.cpu().item()}, gen_images.detach()

    def training_step_D(batch_gen_images, batch_real_images):
        discriminator.train()
        discriminator.zero_grad()

        d_real = discriminator(batch_real_images)
        err_D_real = lossfn_D_real(d_real)

        d_fake = discriminator(batch_gen_images)
        err_D_fake = lossfn_D_fake(d_fake)

        loss_total = 0.5 * (err_D_real + err_D_fake)

        loss_total.backward()
        optimizer_D.step()

        return {'errD_real': err_D_real.cpu().item(),
                'errD_fake': err_D_fake.cpu().item(),
                'errD': loss_total.cpu().item()}

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,), std=(0.5,))
    ])

    dataloader = mnist_loader.get_loader(
        root_path=args.image_dir,
        batch_size=args.batch_size,
        transform=tsfm,
        mode='train')
    print('dataloader', dataloader)

    static_batch_z = torch.zeros((16, args.z_dim)).uniform_(-1.0, 1.0).to(device)

    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'log/'))

    for epoch in range(1, args.num_epochs+1):

        for i, (batch_real, _) in enumerate(dataloader):

            batch_z = torch.zeros(batch_real.shape[0], args.z_dim).uniform_(-1.0, 1.0)
            batch_z_dev = batch_z.to(device)

            batch_real_dev = batch_real.to(device)

            losses_g, gen_images = training_step_G(batch_z_dev)

            losses_d = training_step_D(gen_images, batch_real_dev)

            if i % args.log_interval == 0:
                print('Epoch {:<3d}/{} Iter {:>3d}/{} G: {:.4f} D: {:.4f}'
                      ''.format(epoch, args.num_epochs, i, len(dataloader),
                                losses_g['errG'], losses_d['errD']))

            writer.add_scalar(
                'loss/errG', losses_g['errG'], global_step=epoch)
            writer.add_scalar(
                'loss/errD', losses_d['errD'], global_step=epoch)
            writer.add_scalar(
                'loss/errD_real', losses_d['errD_real'], global_step=epoch)
            writer.add_scalar(
                'loss/errD_fake', losses_d['errD_fake'], global_step=epoch)

        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict()
                },
                os.path.join(
                    args.checkpoint_dir,
                    'model-{}.tch'.format(epoch)
                )
            )

            generator.eval()
            outputs = generator(static_batch_z).detach().cpu()
            outputs = 1.0 - (outputs * 0.5 + 0.5)

            grid_generated = torchvision.utils.make_grid(outputs, nrow=4)

            writer.add_image('images/generated', grid_generated, epoch + 1)


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--image_dir', type=str, required=True,
            help='The path to the training directory')
    parser.add_argument(
            '--z_dim', type=int, required=False, default=50,
            help='The size of the latent (z) dimnesion')
    parser.add_argument(
            '--hidden_dim', type=int, required=False, default=128,
            help='The size of the hidden layer')
    parser.add_argument(
            '--image_dim', type=int, required=False, default=28,
            help='The size (w or h) of the squared gray-scaled image')
    parser.add_argument(
            '--image_channels', type=int, required=False, default=1,
            help='Number of image channels')
    parser.add_argument(
            '--p_drop', type=float, required=False, default=0.5,
            help='The probability of drop for the Dropout layers')

    parser.add_argument(
            '--fully_connected', type=int, required=False, default=1,
            help='Flag to determine fully-connected networks or not')
    parser.add_argument(
            '--nf_generator', type=int, required=False, default=64,
            help='Number of filters for the generator')
    parser.add_argument(
            '--nf_discriminator', type=int, required=False, default=8,
            help='Number of filters for the discriminator')

    # The rest of arguments
    parser.add_argument(
            '--ngpu', type=int, default=1,
            help='Number of GPUs to use')
    parser.add_argument(
            '--gan_loss', type=str, required=True,
            choices=['vanilla', 'wgan'],
            help='The type of GAN loss to sue for training')
    parser.add_argument(
            '--batch_size', type=int, default=128,
            help='Batch size of training')
    parser.add_argument(
            '--num_workers', type=int, default=1,
            help='Number of workers for loading data')
    parser.add_argument(
            '--learning_rate', type=float, default=1e-4,
            help='Learning rate')
    parser.add_argument(
            '--num_epochs', type=int, default=100,
            help='Number of epochs for training the model')
    parser.add_argument(
            '--log_interval', type=int, default=100,
            help='The interval steps to print the loss values')
    parser.add_argument(
            '--beta1', type=float, default=0.5,
            help='Bets for the Adam optimizer')
    parser.add_argument(
            '--checkpoint_dir', type=str, default='/tmp/checkpoints/',
            help='Checkpoint directory for training')

    args = parser.parse_args(argv)

    if args.checkpoint_dir == '/tmp/checkpoints/':
        timestr = time.strftime('%m_%d_%Y-%H_%M_%S')
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, timestr)

    ckpt_path = pathlib.Path(args.checkpoint_dir)
    ckpt_path.mkdir(parents=True)

    return args


if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)
