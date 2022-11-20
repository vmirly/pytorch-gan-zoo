"""
PyTorch GAN Zoo - unittests for FC-networks
Author: Vahid Mirjalili
"""

from ganzoo.misc import ops
from ganzoo.nn_modules import fc_nets
from ganzoo.constants import data_constants
from ganzoo.constants import names
from ganzoo.constants import defaults


def test_fc_networks():

    image_shape = data_constants.IMAGE_SHAPES[names.NAMESTR_MNIST]
    image_dim, image_channels = image_shape[0], image_shape[2]

    classes = [
        (fc_nets.FCSmall_Generator, fc_nets.FCSmall_Discriminator),
        (fc_nets.FCSkipConnect_Generator, fc_nets.FCSkipConnect_Discriminator),
        (fc_nets.FCLarge_Generator, fc_nets.FCLarge_Discriminator)
    ]
    for gen_class, disc_class in classes:
        generator = fc_nets.FCSmall_Generator(
            num_z_units=10,
            num_hidden_units=10,
            output_image_dim=image_dim,
            output_image_channels=image_channels,
            p_drop=0.5)

        input_size = image_dim * image_dim * image_channels
        discriminator = fc_nets.FCSmall_Discriminator(
            input_feature_dim=input_size,
            num_hidden_units=10,
            p_drop=0.5,
            activation=defaults.DISC_ACTIVATIONS['vanilla'])

        # test with uniform z
        sample_z = ops.get_latent_sampler(
            z_dim=10,
            z_distribution=names.NAMESTR_UNIFORM,
            make_4d=False)
        z = sample_z(batch_size=2)

        fake = generator(z)
        assert fake.shape == (2, 1, image_dim, image_dim)

        d_fake = discriminator(fake)
        assert d_fake.shape == (2, 1)

        # test with normal z
        sample_z = ops.get_latent_sampler(
            z_dim=10,
            z_distribution=names.NAMESTR_UNIFORM,
            make_4d=False)
        z = sample_z(batch_size=2)

        fake = generator(z)
        assert fake.shape == (2, 1, image_dim, image_dim)

        d_fake = discriminator(fake)
        assert d_fake.shape == (2, 1)
