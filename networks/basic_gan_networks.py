import torch.nn as nn

from ops import torch_ops


def make_generator(
        num_z_units,
        num_hidden_units,
        output_image_dim,
        p_drop=0.5):

    output_image_size = output_image_dim * output_image_dim

    model = nn.Sequential(
        nn.Linear(num_z_units, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, output_image_size),
        nn.Tanh(),
        torch_ops.Reshape((1, output_image_dim, output_image_dim)))

    return model


def make_discriminator(
        input_image_dim,
        num_hidden_units,
        p_drop=0.5):

    input_image_size = input_image_dim * input_image_dim

    model = nn.Sequential(
        torch_ops.Flatten(),
        nn.Linear(input_image_size, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, 1),
        nn.Sigmoid())
        #torch_ops.Flatten())

    return model


def make_conv_generator(
        num_z_units,
        num_filters):

    nf1 = num_filters
    nf2 = num_filters // 2
    nf3 = num_filters // 4
    nf4 = num_filters // 8

    model = nn.Sequential(
        nn.Linear(num_z_units, 3136, bias=False),
        nn.BatchNorm1d(num_features=3136),
        nn.LeakyReLU(inplace=True, negative_slope=0.0001),
        torch_ops.Reshape(new_shape=(64, 7, 7)),
            
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
            out_channels=1,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=0,
            bias=False),

        nn.Tanh())

    return model


def make_conv_discriminator():

    model = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=8,
            padding=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            bias=False),

        nn.BatchNorm2d(num_features=8),
        nn.LeakyReLU(inplace=True, negative_slope=0.0001), 
            
        nn.Conv2d(
            in_channels=8,
            out_channels=32,
            padding=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            bias=False),

        nn.BatchNorm2d(num_features=32),
        nn.LeakyReLU(inplace=True, negative_slope=0.0001), 
            
        torch_ops.Flatten(),

        nn.Linear(7*7*32, 1),
        nn.Sigmoid())

    return model
