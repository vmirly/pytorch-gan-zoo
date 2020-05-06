import torch
import torch.nn as nn

from networks import basic_gan_networks as basic_nets


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


class ConditionalConvGAN(nn.Module):

    def __init__(
            self,
            num_z_units,
            num_cond_vals,
            image_dim,
            image_channels,
            g_num_filters,
            d_num_filters,
            use_fusion_layer):

        super(ConditionalConvGAN, self).__init__()
        self.image_dim = image_dim
        self.num_cond_vals = num_cond_vals
        self.use_fusion_layer = use_fusion_layer

        self.generator = basic_nets.make_conv_generator(
            num_z_units=num_z_units + num_cond_vals,
            num_filters=g_num_filters,
            output_image_dim=image_dim,
            output_image_channels=image_channels)

        if self.use_fusion_layer:
            num_inp_channels = image_channels + 1
        else:
            num_inp_channels = image_channels + num_cond_vals

        self.discriminator = basic_nets.make_conv_discriminator(
            image_dim=image_dim,
            num_inp_channels=num_inp_channels,
            num_filters=d_num_filters)

        if self.use_fusion_layer:
            # A non-trainable layer for projecting conditional labels
            # Only defined when use_fusion_layer is True
            self.c_projector = nn.Linear(
                in_features=num_cond_vals,
                out_features=image_dim * image_dim,
                bias=False)
            self.c_projector.weight.requires_grad = False

    def gen_forward(self, z, c):
        inputs = torch.cat([z, c], dim=1)

        return self.generator(inputs)

    def disc_forward(self, imgs, c):
        if not self.use_fusion_layer:
            c = c.view(-1, self.num_cond_vals, 1, 1)
            c = c.repeat((1, 1, self.image_dim, self.image_dim))
        else:
            c = self.c_projector(c).view(-1, 1, self.image_dim, self.image_dim)

        inputs = torch.cat([imgs, c], dim=1)

        return self.discriminator(inputs)
