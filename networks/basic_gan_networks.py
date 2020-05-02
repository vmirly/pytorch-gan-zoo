import torch.nn as nn

from ops import torch_ops


def make_generator(
        num_z_units,
        num_hidden_units,
        output_image_size,
        p_drop=0.5):

    model = nn.Sequential(
        nn.Linear(num_z_units, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, output_image_size),
        nn.Tanh())

    return model


def make_discriminator(
        input_image_size,
        num_hidden_units,
        p_drop=0.5):

    model = nn.Sequential(
        nn.Linear(input_image_size, num_hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=p_drop),
        nn.Linear(num_hidden_units, 1),
        nn.Sigmoid(),
        torch_ops.Flatten())

    return model
