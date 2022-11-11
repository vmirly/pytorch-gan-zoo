"""
PyTorch GAN Zoo - unittests for FC-networks
Author: Vahid Mirjalili
"""

from ganzoo.misc import ops
from ganzoo.networks import fc_nets
from ganzoo.constants import data_constants
from ganzoo.constants import network_constants


def test_fc_networks():

    image_dim = data_constants.IMAGE_DIMS['mnist']
    image_channels = data_constants.IMAGE_CHANNELS['mnist']

    generator = fc_nets.FC_Generator(
        num_z_units=10,
        num_hidden_units=10,
        output_image_dim=image_dim,
        output_image_channels=image_channels,
        p_drop=0.5)

    input_size = image_dim * image_dim * image_channels
    discriminator = fc_nets.FC_Discriminator(
        input_feature_dim=input_size,
        num_hidden_units=10,
        p_drop=0.5,
        activation=network_constants.DISC_ACTIVATIONS['vanilla'])

    # test with uniform z
    sample_z = ops.get_latent_sampler(
        z_dim=10,
        z_distribution='uniform',
        make_4d=False)
    z = sample_z(batch_size=2)

    fake = generator(z)
    assert fake.shape == (2, 1, image_dim, image_dim)

    d_fake = discriminator(fake)
    assert d_fake.shape == (2, 1)

    # test with normal z
    sample_z = ops.get_latent_sampler(
        z_dim=10,
        z_distribution='normal',
        make_4d=False)
    z = sample_z(batch_size=2)

    fake = generator(z)
    assert fake.shape == (2, 1, image_dim, image_dim)

    d_fake = discriminator(fake)
    assert d_fake.shape == (2, 1)
