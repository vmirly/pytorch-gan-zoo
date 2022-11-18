"""
PyTorch GAN Zoo -- script to run PL-basic-FC-GAN
Author: Vahid Mirjalili
"""
import sys
import argparse
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms as T

from ganzoo.lit_modules import basic_fc_gan
from ganzoo.lit_modules import lit_data_vision
from ganzoo.constants import names
from ganzoo.constants import defaults


def parse(argv):
    parser = argparse.ArgumentParser(__file__)
    
    parser.add_argument(
        '--dataset_name', type=str, required=True,  # TODO: add custom-folder 
        choices=names.DOWNLOADABLE_DATASETS)

    parser.add_argument(
        '--z_dim', type=int, required=False,
        default=defaults.Z_DIM)

    parser.add_argument(
        '--z_distribution', type=str,
        default=names.NAMESTR_UNIFORM,
        choices=names.LIST_DISTRBUTIONS)

    parser.add_argument(
        '--network_type', type=str,
        default=names.NAMESTR_FCSKIP,
        choices=names.LIST_FC_NETWORKS)

    parser.add_argument(
        '--num_hidden_units', type=int,
        default=defaults.FC_HIDDEN_UNITS)

    parser.add_argument(
        '--p_drop', type=float, required=False,
        default=defaults.DROP_PROBA)

    parser.add_argument(
        '--num_epochs', type=int, required=False,
        default=defaults.NUM_EPOCHS)

    args = parser.parse_args()
    return args


def main(args):

    model = basic_fc_gan.LitBasicGANFC(
        num_z_units=args.z_dim,
        z_distribution=args.z_distribution,
        num_hidden_units=args.num_hidden_units,
        image_dim=28, image_channels=1, p_drop=args.p_drop,
        lr=0.001, beta1=0.5, beta2=0.9,
        network_type=args.network_type)

    trsfm = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    dm = lit_data_vision.LitVisionDataset(
        dataset_name=args.dataset_name,
        transform=trsfm,
        splits=(0.9, 0.1),
        batch_size=32)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.num_epochs,
    )
    trainer.fit(model, dm)


if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)
