"""
PyTorch GAN Zoo -- script to run PL-basic-FC-GAN
Author: Vahid Mirjalili
"""
import sys
import argparse
import torch
import pytorch_lightning as pl

from ganzoo.pl_modules import basic_gan_fc
from ganzoo.misc import ops


def parse(argv):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument(
        '--z_distribution', type=str, default='uniform',
        choices=['uniform', 'normal'])
    parser.add_argument(
        '--num_hidden_units', type=int, default=64)
    parser.add_argument(
        '--p_drop', type=float, default=0.2)
    parser.add_argument(
        '--num_epochs', type=int, default=100)

    args = parser.parse_args()
    return args


def main(args):
    z_sampler = ops.get_latent_sampler(
        z_dim=args.z_dim,
        z_distribution=args.z_distribution, make_4d=False)

    model = basic_gan_fc.PLBasicGANFC(
        num_z_units=args.z_dim,
        num_hidden_units=args.num_hidden_units,
        image_dim=28, image_channels=1, p_drop=args.p_drop,
        lr=0.001, beta1=0.5, beta2=0.9,
        z_sampler=z_sampler)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.num_epochs,
    )
    trainer.fit(model)


if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)
