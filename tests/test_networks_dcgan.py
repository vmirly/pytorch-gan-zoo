"""
PyTorch GAN Zoo - unittests for DCGAN-networks
Author: Vahid Mirjalili
"""

from ganzoo.misc import ops
from ganzoo.nn_modules import dcgan_nets
from ganzoo.constants import data_constants
from ganzoo.constants import names
from ganzoo.constants import defaults


def test_dcgan_networks():

    image_shape = data_constants.IMAGE_SHAPES[names.NAMESTR_CIFAR10]
    image_dim, image_channels = image_shape[0], image_shape[2]

    generator = dcgan_nets.DCGAN_Generator(
        num_z_units=10,
        num_conv_filters=2,
        output_image_dim=image_dim,
        output_image_channels=image_channels)

    discriminator = dcgan_nets.DCGAN_Discriminator(
        image_dim=image_dim,
        image_channels=image_channels,
        num_conv_filters=2,
        activation=defaults.DISC_ACTIVATIONS['vanilla'])

    # test with uniform z
    sample_z = ops.get_latent_sampler(
        z_dim=10,
        z_distribution=names.NAMESTR_UNIFORM,
        make_4d=True)
    z = sample_z(batch_size=2)

    fake = generator(z)
    assert fake.shape == (2, image_channels, image_dim, image_dim)

    d_fake = discriminator(fake)
    assert d_fake.shape == (2,)

    # test with normal z
    sample_z = ops.get_latent_sampler(
        z_dim=10,
        z_distribution=names.NAMESTR_UNIFORM,
        make_4d=True)
    z = sample_z(batch_size=2)

    fake = generator(z)
    assert fake.shape == (2, image_channels, image_dim, image_dim)

    d_fake = discriminator(fake)
    assert d_fake.shape == (2,)
