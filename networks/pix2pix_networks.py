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

        # Contractive layers:
        self.conv1a = nn.Conv2d(
            in_channels=n_inp_channels, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters*2,
            kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(
            in_channels=n_filters*2, out_channels=n_filters*2,
            kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(
            in_channels=n_filters*2, out_channels=n_filters*4,
            kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(
            in_channels=n_filters*4, out_channels=n_filters*4,
            kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(
            in_channels=n_filters*4, out_channels=n_filters*8,
            kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(
            in_channels=n_filters*8, out_channels=n_filters*8,
            kernel_size=3, stride=1, padding=1)

        self.conv5a = nn.Conv2d(
            in_channels=n_filters*8, out_channels=n_filters*16,
            kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(
            in_channels=n_filters*16, out_channels=n_filters*16,
            kernel_size=3, stride=1, padding=1)

        # Expansive layers:
        self.t_conv6 = nn.ConvTranspose2d(
            in_channels=n_filters*16, out_channels=n_filters*8,
            kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv6a = nn.Conv2d(
            in_channels=n_filters*16, out_channels=n_filters*8,
            kernel_size=3, stride=1, padding=1)
        self.conv6b = nn.Conv2d(
            in_channels=n_filters*8, out_channels=n_filters*8,
            kernel_size=3, stride=1, padding=1)

        self.t_conv7 = nn.ConvTranspose2d(
            in_channels=n_filters*8, out_channels=n_filters*4,
            kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv7a = nn.Conv2d(
            in_channels=n_filters*8, out_channels=n_filters*4,
            kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(
            in_channels=n_filters*4, out_channels=n_filters*4,
            kernel_size=3, stride=1, padding=1)

        self.t_conv8 = nn.ConvTranspose2d(
            in_channels=n_filters*4, out_channels=n_filters*2,
            kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv8a = nn.Conv2d(
            in_channels=n_filters*4, out_channels=n_filters*2,
            kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(
            in_channels=n_filters*2, out_channels=n_filters*2,
            kernel_size=3, stride=1, padding=1)

        self.t_conv9 = nn.ConvTranspose2d(
            in_channels=n_filters*2, out_channels=n_filters,
            kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv9a = nn.Conv2d(
            in_channels=n_filters*2, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1)
        self.conv9b = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1)

        # Last conv layer avec 1x1 kernel
        self.conv_out = nn.Conv2d(
            in_channels=n_filters, out_channels=n_out_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Conctractive layers:
        x1 = F.leaky_relu(self.conv1a(x))
        x1 = F.leaky_relu(self.conv1b(x1))
        x1pool = F.max_pool2d(x1, 2)

        x2 = F.leaky_relu(self.conv2a(x1pool))
        x2 = F.leaky_relu(self.conv2b(x2))
        x2pool = F.max_pool2d(x2, 2)

        x3 = F.leaky_relu(self.conv3a(x2pool))
        x3 = F.leaky_relu(self.conv3b(x3))
        x3pool = F.max_pool2d(x3, 2)

        x4 = F.leaky_relu(self.conv4a(x3pool))
        x4 = F.leaky_relu(self.conv4b(x4))
        x4pool = F.max_pool2d(x4, 2)

        x5 = F.leaky_relu(self.conv5a(x4pool))
        x5 = F.leaky_relu(self.conv5b(x5))

        # Expansive layers:
        x6 = F.leaky_relu(self.t_conv6(x5))
        x6 = torch.cat([x6, x4], axis=1)
        x6 = F.leaky_relu(self.conv6a(x6))
        x6 = F.leaky_relu(self.conv6b(x6))

        x7 = F.leaky_relu(self.t_conv7(x6))
        x7 = torch.cat([x7, x3], axis=1)
        x7 = F.leaky_relu(self.conv7a(x7))
        x7 = F.leaky_relu(self.conv7b(x7))

        x8 = F.leaky_relu(self.t_conv8(x7))
        x8 = torch.cat([x8, x2], axis=1)
        x8 = F.leaky_relu(self.conv8a(x8))
        x8 = F.leaky_relu(self.conv8b(x8))

        x9 = F.leaky_relu(self.t_conv9(x8))
        x9 = torch.cat([x9, x1], axis=1)
        x9 = F.leaky_relu(self.conv9a(x9))
        x9 = F.leaky_relu(self.conv9b(x9))

        out = self.conv_out(x9)

        return torch.tanh(out)


class Pix2PixDiscriminator(nn.Module):

    def __init__(
            self,
            n_inp_channels=6,
            n_filters=64):

        super(Pix2PixDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_inp_channels, out_channels=n_filters,
            kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters*2,
            kernel_size=5, stride=2, padding=2)

        self.conv3 = nn.Conv2d(
            in_channels=n_filters*2, out_channels=n_filters*4,
            kernel_size=5, stride=1, padding=2)

        self.conv4 = nn.Conv2d(
            in_channels=n_filters*4, out_channels=n_filters*8,
            kernel_size=5, stride=2, padding=2)

        self.conv5 = nn.Conv2d(
            in_channels=n_filters*8, out_channels=1,
            kernel_size=5, stride=1, padding=2)

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
