
"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""
import argparse
from ganzoo.constants import names
from ganzoo.constants import defaults


def parse_basicfc_train_opts(argv):
    parser = argparse.ArgumentParser(__file__)
    
    parser.add_argument(
        '--dataset_name', type=str, required=True,  # TODO: add custom-folder 
        choices=names.DOWNLOADABLE_DATASETS + [names.NAMESTR_CELEBA])

    parser.add_argument(
        '--z_dim', type=int, required=False,
        default=defaults.Z_DIM)

    parser.add_argument(
        '--z_distribution', type=str,
        default=names.NAMESTR_UNIFORM,
        choices=names.LIST_DISTRBUTIONS)

    parser.add_argument(
        '--num_conv_filters', type=int,
        default=defaults.NUM_CONV_KERNELS)

    parser.add_argument(
        '--batch_size', type=int,
        default=defaults.BATCH_SIZE)

    parser.add_argument(
        '--desired_image_size', type=int, required=False,
        choices=[32, 64], default=32)

    parser.add_argument(
        '--num_epochs', type=int, required=False,
        default=defaults.NUM_EPOCHS)

    parser.add_argument(
        '--loss_type', type=str, required=False,
        choices=['vanilla', 'wgan', 'wgan-gp', 'wgan-lp'],
        default='vanilla')

    args = parser.parse_args()
    return args
