import torch.nn as nn

from ops import torch_ops


def make_generator(
        num_z_units,
        num_hidden_units,
        output_image_dim,
        output_image_channels,
        p_drop=0.5):

    output_image_size = (
        output_image_channels * output_image_dim * output_image_dim)

    model = nn.Sequential(
        nn.Linear(num_z_units, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, output_image_size),
        nn.Tanh(),
        torch_ops.Reshape(
            (output_image_channels,
             output_image_dim,
             output_image_dim)))

    return model


def make_discriminator(
        input_feature_dim,
        num_hidden_units,
        p_drop=0.5):

    model = nn.Sequential(
        torch_ops.Flatten(),
        nn.Linear(input_feature_dim, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, 1),
        nn.Sigmoid())

    return model


def make_conv_generator(
        num_z_units,
        num_filters,
        output_image_dim,
        output_image_channels):

    nf1 = num_filters
    nf2 = num_filters // 2
    nf3 = num_filters // 4
    nf4 = num_filters // 8

    feature_dim = output_image_dim // 4

    n_hidden_units = nf1 * feature_dim * feature_dim

    model = nn.Sequential(
        nn.Linear(num_z_units, n_hidden_units, bias=False),
        nn.BatchNorm1d(num_features=n_hidden_units),
        nn.LeakyReLU(inplace=True, negative_slope=0.0001),
        torch_ops.Reshape(new_shape=(nf1, feature_dim, feature_dim)),

        nn.ConvTranspose2d(
            in_channels=nf1,
            out_channels=nf2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False),
        nn.BatchNorm2d(num_features=nf2),
        nn.LeakyReLU(inplace=True, negative_slope=0.0001),
            
        nn.ConvTranspose2d(
            in_channels=nf2,
            out_channels=nf3,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False),
        nn.BatchNorm2d(num_features=nf3),
        nn.LeakyReLU(inplace=True, negative_slope=0.0001),
            
        nn.ConvTranspose2d(
            in_channels=nf3,
            out_channels=nf4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=False),
        nn.BatchNorm2d(num_features=nf4),
        nn.LeakyReLU(inplace=True, negative_slope=0.0001),
            
        nn.ConvTranspose2d(
            in_channels=nf4,
            out_channels=output_image_channels,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=0,
            bias=False),

        nn.Tanh())

    return model


def make_conv_discriminator(
        image_dim,
        num_inp_channels,
        num_filters):

    nf1 = num_filters
    nf2 = num_filters * 4

    feature_dim = image_dim // 4

    model = nn.Sequential(
        nn.Conv2d(
            in_channels=num_inp_channels,
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
            
        torch_ops.Flatten(),

        nn.Linear(nf2 * feature_dim * feature_dim, 1),
        nn.Sigmoid())

    return model
