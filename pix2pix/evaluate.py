import os
import pathlib
import time
import sys
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from networks import pix2pix_networks as p2p_net
from ops.data_ops import convert_tensor2image
from utils.dataloader_paired import PairedImg2ImgDataset


def main(args):
    print(args)
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.ngpu > 0)
        else "cpu")
    print('Device:', device)

    os.makedirs(args.output_dir, exist_ok=True)

    generator = p2p_net.Pix2PixGenerator(n_filters=args.nf).to(device)
    saved_checkpoint = torch.load(args.model_path)
    generator.load_state_dict(saved_checkpoint['model_state_dict'])
    generator.eval()

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))
    ])

    dataset = PairedImg2ImgDataset(
        image_dir=args.input_dir,
        paired_transform=None,
        transform=tsfm,
        mode='eval')

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    for i, (batch_x, batch_y, batch_f) in enumerate(dataloader):

        if not args.y2x:
            batch_x_dev = batch_x.to(device)
            batch_y_dev = batch_y.to(device)
        else:  # swap the x & y
            batch_x_dev = batch_y.to(device)
            batch_y_dev = batch_x.to(device)

        outputs = generator(batch_x_dev)

        for f, out, x, y in zip(batch_f, outputs, batch_x_dev, batch_y_dev):
            img_out = convert_tensor2image(
                out.detach().cpu(),
                unnormalize=True)
            img_out.save(
                os.path.join(args.output_dir, f))

            img_x = convert_tensor2image(
                x.detach().cpu(),
                unnormalize=True)
            img_x.save(
                os.path.join(args.output_dir, f[:-4] + '-inp.jpg'))

            img_y = convert_tensor2image(
                y.detach().cpu(),
                unnormalize=True)
            img_y.save(
                os.path.join(args.output_dir, f[:-4] + '-target.jpg'))

    return outputs


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='The path to the input image directory')
    parser.add_argument(
        '--y2x', type=int, default=0,
        help='Flag for transforming y to x (instead of x to y)')
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='The path to the saved model')
    parser.add_argument(
        '--ngpu', type=int, default=1,
        help='Number of GPUs to use')
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size of training')
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='Number of workers for loading data')
    parser.add_argument(
        '--num_epochs', type=int, default=20,
        help='Number of epochs for training the model')
    parser.add_argument(
        '--nf', type=int, required=True,
        help='Number of filters for the generator network')
    parser.add_argument(
        '--output_dir', type=str,
        default='/tmp/pix2pix-outputs/',
        help='Bets for the Adam optimizer')

    args = parser.parse_args()

    args.y2x = bool(args.y2x)

    if args.output_dir == '/tmp/pix2pix-outputs/':
        timestr = time.strftime('%m_%d_%Y-%H_%M_%S')
        args.output_dir = os.path.join(args.output_dir, timestr)

    ckpt_path = pathlib.Path(args.output_dir)
    ckpt_path.mkdir(parents=True)

    return args


if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)
