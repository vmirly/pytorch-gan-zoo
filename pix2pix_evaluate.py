import sys
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from networks import pix2pix_networks as p2p_net
from ops.dataloader_paired import PairedImg2ImgDataset


def main(args):
    print(args)
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.ngpu > 0)
        else "cpu")
    print('Device:', device)

    generator = p2p_net.Pix2PixGenerator().to(device)

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))
    ])

    dataset = PairedImg2ImgDataset(
        image_dir=args.eval_path,
        paired_transform=None,
        transform=tsfm)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = generator(batch_x)

    return batch_x, batch_y, output


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, required=True,
                        help='The path to the training directory')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size of training')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for loading data')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs for training the model')
    parser.add_argument('--output_dir', type=str,
        default='/tmp/pix2pix-outputs/'
                        help='Bets for the Adam optimizer')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)
