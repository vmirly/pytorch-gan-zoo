import torch
import torch.nn as nn
import torch.nn.functional as F


class Pix2PixGenerator(nn.Module):
    def __init__(
            self,
            n_inp_channels=3,
            n_out_channels=3,
            n_filters=64):

        super(Pix2PixGenerator, self).__init__()

        # Encoder layers:
        self.conv1 = nn.Conv2d(
            in_channels=n_inp_channels, out_channels=n_filters,
            kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters*2,
            kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=n_filters*2, out_channels=n_filters*4,
            kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.Conv2d(
            in_channels=n_filters*4, out_channels=n_filters*8,
            kernel_size=4, stride=2, padding=1)

        # Decoder layers:
        self.conv5 = nn.ConvTranspose2d(
            in_channels=n_filters*8, out_channels=n_filters*4,
            kernel_size=4, stride=2, padding=1, output_padding=0)

        self.conv6 = nn.ConvTranspose2d(
            in_channels=n_filters*4, out_channels=n_filters*2,
            kernel_size=4, stride=2, padding=1, output_padding=0)

        self.conv7 = nn.ConvTranspose2d(
            in_channels=n_filters*2, out_channels=n_filters,
            kernel_size=4, stride=2, padding=1, output_padding=0)

        self.conv8 = nn.ConvTranspose2d(
            in_channels=n_filters, out_channels=n_filters//2,
            kernel_size=4, stride=2, padding=1, output_padding=0)

        # Last conv layer avec 1x1 kernel
        self.conv_out = nn.Conv2d(
            in_channels=n_filters//2, out_channels=n_out_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))

        x = self.conv_out(x)

        return torch.tanh(x)


class Pix2PixDiscriminator(nn.Module):

    def __init__(
            self,
            n_inp_channels=6,
            n_filters=64):

        super(Pix2PixDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_inp_channels,
            out_channels=n_filters,
            kernel_size=5,
            stride=1,
            padding=2)

        self.conv2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters*2,
            kernel_size=5,
            stride=2,
            padding=2)

        self.conv3 = nn.Conv2d(
            in_channels=n_filters*2,
            out_channels=n_filters*4,
            kernel_size=5,
            stride=1,
            padding=2)

        self.conv4 = nn.Conv2d(
            in_channels=n_filters*4,
            out_channels=n_filters*8,
            kernel_size=5,
            stride=2,
            padding=2)

        self.conv5 = nn.Conv2d(
            in_channels=n_filters*8,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)

        return x
