"""
PyTorch GAN Zoo -- script to run PL-basic-FC-GAN
Author: Vahid Mirjalili
"""
import sys
import logging
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms as T

from ganzoo.lit_modules import basic_fc_gan
from ganzoo.lit_modules import lit_data_vision
from ganzoo.misc import utils
from run.basic_gan import parser_fc as parser


def main(args):

    image_shape, msg = utils.get_dataset_props(args.dataset_name)
    if msg:
        return msg
    image_h, image_w, image_c = image_shape
    if image_h != image_w:
        return 'image is not squared: image_dim is not defined'
    image_dim = image_h
    image_channels = image_c

    model = basic_fc_gan.LitBasicGANFC(
        num_z_units=args.z_dim,
        z_distribution=args.z_distribution,
        num_hidden_units=args.num_hidden_units,
        image_dim=image_dim, image_channels=image_channels,
        p_drop=args.p_drop,
        lr=0.001, beta1=0.5, beta2=0.9,
        network_type=args.network_type,
        loss_type=args.loss_type)

    trsfm = T.Compose([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])

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
    args = parser.parse_basicfc_train_opts(sys.argv[1:])
    msg = main(args)
    if msg:
        logging.error(msg)
        sys.exit(1)
