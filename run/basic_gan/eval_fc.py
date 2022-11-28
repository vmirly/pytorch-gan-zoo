"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""
import os
import sys
import argparse
from PIL import Image
import torch
import torchvision
import pytorch_lightning as pl

from ganzoo.lit_modules import basic_fc_gan


def parse(argv):
    parser = argparse.ArgumentParser(__file__)

    parser.add_argument(
        '--checkpoint_path', type=str, required=True)
    parser.add_argument(
        '--output_dir', type=str, required=True)

    args = parser.parse_args()

    return args


def main(args):
    model = basic_fc_gan.LitBasicGANFC.load_from_checkpoint(
        args.checkpoint_path)
    print(model)

    os.makedirs(args.output_dir, exist_ok=True)

    bz = model.z_sampler(batch_size=64)
    gen_imgs = model(bz)
    grid = torchvision.utils.make_grid(
        gen_imgs, normalize=True, value_range=(-1.0, 1.0))
    image = (grid * 255).numpy().transpose(1, 2, 0).astype('uint8')
    image = Image.fromarray(image)
    image.save(os.path.join(args.output_dir, 'generated-images.png'))


if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)
