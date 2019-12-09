import sys
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from ops import constants
from networks import pix2pix_networks as p2p_net
from ops.dataloader_paired import PairedImg2ImgDataset
from ops import paired_image_transforms


def main(args):
    print(args)
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.ngpu > 0)
        else "cpu")
    print('Device:', device)

    lossfn_rec = constants.RECONSTRUCTION_LOSS[args.rec_loss]
    lossfn_G, lossfn_D_real, lossfn_D_fake = constants.GAN_LOSS[args.gan_loss]

    generator = p2p_net.Pix2PixGenerator().to(device)
    discriminator = p2p_net.Pix2PixDiscriminator().to(device)

    optimizer_D = optim.Adam(
        generator.parameters(), lr=args.learning_rate,
        betas=(args.beta1, 0.999))
    optimizer_G = optim.Adam(
        discriminator.parameters(), lr=args.learning_rate,
        betas=(args.beta1, 0.999))

    def training_step_G(batch_x, batch_y):
        generator.zero_grad()

        output = generator(batch_x)

        rec_loss = lossfn_rec(input=output, target=batch_y)

        input_fake = torch.cat([batch_x, output], axis=1)
        d_fake = discriminator(input_fake)
        err_G = lossfn_G(d_fake)

        loss_g = rec_loss + err_G*args.lambda_gan

        loss_g.backward()
        optimizer_G.step()

        return {'rec': rec_loss.cpu().item(),
                'errG': err_G.cpu().item()}

    def training_step_D(batch_x, batch_y):
        discriminator.zero_grad()

        input_real = torch.cat([batch_x, batch_y], axis=1)
        d_real = discriminator(input_real)
        err_D_real = lossfn_D_real(d_real)
        err_D_real.backward()

        output = generator(batch_x)

        input_fake = torch.cat([batch_x, output], axis=1)
        d_fake = discriminator(input_fake)
        err_D_fake = lossfn_D_fake(d_fake)
        err_D_fake.backward()

        err_D = err_D_real + err_D_fake

        # err_D.backward()
        optimizer_D.step()

        return {'errD_real': err_D_real.cpu().item(),
                'errD_fake': err_D_fake.cpu().item(),
                'errD': err_D.cpu().item()}

    paired_tsfm = transforms.Compose([
        paired_image_transforms.RandomPairedHFlip(prob=1.0),
    ])
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))
    ])

    dataset = PairedImg2ImgDataset(
        image_dir=args.train_path,
        paired_transform=paired_tsfm,
        transform=tsfm)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    for epoch in range(args.num_epochs):

        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            losses_g = training_step_G(batch_x, batch_y)

            losses_d = training_step_D(batch_x, batch_y)

            if i % args.log_interval == 0:
                print('Epoch {}/{} Iter {} Rec: {:.3f} G: {:.3f} D: {}'
                      ''.format(epoch, args.num_epochs, i,
                                losses_g['rec'], losses_g['errG'],
                                losses_d['errD']))

    return losses_g, losses_d


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help='The path to the training directory')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gan_loss', type=str, required=True,
                        choices=['vanilla', 'wgan'],
                        help='The type of GAN loss to sue for training')
    parser.add_argument('--rec_loss', required=True,
                        choices=['l1', 'l2', 'ce'],
                        help='The type of reconstruction loss')
    parser.add_argument('--lambda_gan', type=float, default=1.0,
                        help='Lambda for the weight of gan loss')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size of training')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for loading data')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs for training the model')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='The interval steps to print the loss values')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Bets for the Adam optimizer')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)