import torch
import torch.nn as nn

from networks import basic_gan_networks as basic_nets
from ops import torch_ops


class ConditionalFullyConnectedGAN(nn.Module):
    def __init__(
            self,
            num_z_units,
            num_cond_vals,
            num_hidden_units,
            image_dim,
            image_channels,
            p_drop):
        super(ConditionalFullyConnectedGAN, self).__init__()

        self.generator = basic_nets.make_generator(
            num_z_units=num_z_units + num_cond_vals,
            num_hidden_units=num_hidden_units,
            output_image_dim=image_dim,
            output_image_channels=image_channels,
            p_drop=p_drop)

        self.discriminator = basic_nets.make_discriminator(
            input_feature_dim=image_channels * image_dim * image_dim + num_cond_vals,
            num_hidden_units=num_hidden_units,
            p_drop=p_drop)

    def gen_forward(self, z, c):
        inputs = torch.cat([z, c], dim=1)

        return self.generator(inputs)

    def disc_forward(self, imgs, c):
        imgs = imgs.view(imgs.shape[0], -1)
        inputs = torch.cat([imgs, c], dim=1)

        return self.discriminator(inputs)


class LateFusionDiscriminator(nn.Module):

    def __init__(
            self,
            image_dim,
            image_channels,
            num_filters,
            num_cond_vals):
        super(LateFusionDiscriminator, self).__init__()

        nf1 = num_filters
        nf2 = num_filters * 4

        feature_dim = image_dim // 4

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=nf1,
                padding=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                bias=False),

            nn.BatchNorm2d(num_features=nf1),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),

            nn.Conv2d(
                in_channels=nf1,
                out_channels=nf2,
                padding=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                bias=False),

            nn.BatchNorm2d(num_features=nf2),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),

            torch_ops.Flatten())

        # Final FC layer after feature fusion
        self.fc = nn.Sequential(
            nn.Linear(nf2 * feature_dim * feature_dim + num_cond_vals, 1),

            nn.Sigmoid())

    def forward(self, x, y):
        x = self.feature_extractor(x)

        x = torch.cat([x, y], dim=1)

        return self.fc(x)


class ConditionalConvGAN(nn.Module):

    def __init__(
            self,
            num_z_units,
            num_cond_vals,
            image_dim,
            image_channels,
            g_num_filters,
            d_num_filters):

        super(ConditionalConvGAN, self).__init__()
        self.image_dim = image_dim
        self.num_cond_vals = num_cond_vals

        self.generator = basic_nets.make_conv_generator(
            num_z_units=num_z_units + num_cond_vals,
            num_filters=g_num_filters,
            output_image_dim=image_dim,
            output_image_channels=image_channels)

        # self.discriminator = basic_nets.make_conv_discriminator(
        #    image_dim=image_dim,
        #    num_inp_channels=image_channels + num_cond_vals,
        #    num_filters=d_num_filters)
        self.discriminator = LateFusionDiscriminator(
            image_dim=image_dim,
            image_channels=image_channels,
            num_filters=d_num_filters,
            num_cond_vals=num_cond_vals)

    def gen_forward(self, z, c):
        inputs = torch.cat([z, c], dim=1)

        return self.generator(inputs)

    def disc_forward(self, imgs, c):
        # c = c.view(-1, self.num_cond_vals, 1, 1)
        # c = c.repeat((1, 1, self.image_dim, self.image_dim))

        # inputs = torch.cat([imgs, c], dim=1)

        # return self.discriminator(inputs)
        return self.discriminator(imgs, c)
