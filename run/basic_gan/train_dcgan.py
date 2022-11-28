"""
PyTorch GAN Zoo -- script to train DCGAN using pytorch-lightning
Author: Vahid Mirjalili
"""

import sys
import logging
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms as T

from ganzoo.lit_modules import dcgan
from ganzoo.lit_modules import lit_data_vision
from ganzoo.misc import utils
from run.basic_gan import parser_dcgan as parser


def main(args):

    image_shape, msg = utils.get_dataset_props(args.dataset_name)
    if msg:
        return msg
    image_h, image_w, image_c = image_shape

    if image_h != image_w:
        if max([image_h, image_w]) > 32 and min([image_h, image_w]) < 32:
            return 'Unexpected data shape!'

    image_channels = image_c

    model = dcgan.LitDCGAN(
        num_z_units=args.z_dim,
        z_distribution=args.z_distribution,
        num_conv_filters=args.num_conv_filters,
        image_dim=args.desired_image_size,
        image_channels=image_channels,
        lr=0.001, beta1=0.5, beta2=0.9,
        loss_type=args.loss_type
    )

    if image_channels == 3:
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)
    else:
        norm_mean, norm_std = 0.5, 0.5
    trsfm = T.Compose([
        T.Resize(args.desired_image_size),
        T.CenterCrop(args.desired_image_size),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    dm = lit_data_vision.LitVisionDataset(
        dataset_name=args.dataset_name,
        transform=trsfm,
        splits=(0.9, 0.1),
        batch_size=args.batch_size)

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
