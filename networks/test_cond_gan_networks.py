import unittest
import torch
import numpy as np

from networks import cond_gan_networks as nets


class TestCondGANNetworks(unittest.TestCase):

    def test_fc_networks(self):

        cond_gan = nets.ConditionalFullyConnectedGAN(
            num_z_units=10,
            num_cond_vals=2,
            num_hidden_units=32,
            image_dim=28,
            image_channels=1,
            p_drop=0.5)

        z = np.random.normal(size=(1, 10)).astype(np.float32)
        z = torch.tensor(z)

        c = np.random.uniform(size=(1, 2)).astype(np.float32)
        c = torch.tensor(c)

        gen_output = cond_gan.gen_forward(z, c)
        self.assertTrue(np.array_equal(gen_output.shape, (1, 1, 28, 28)))

        disc_output = cond_gan.disc_forward(gen_output, c)
        self.assertTrue(np.array_equal(disc_output.shape, (1, 1)))

    def test_conv_networks_no_fusion(self):

        cond_gan = nets.ConditionalConvGAN(
            num_z_units=10,
            num_cond_vals=2,
            image_dim=28,
            image_channels=1,
            g_num_filters=64,
            d_num_filters=8,
            use_fusion_layer=False)

        z = np.random.normal(size=(4, 10)).astype(np.float32)
        z = torch.tensor(z)

        c = np.random.uniform(size=(4, 2)).astype(np.float32)
        c = torch.tensor(c)

        gen_output = cond_gan.gen_forward(z, c)
        self.assertTrue(np.array_equal(gen_output.shape, (4, 1, 28, 28)))

        disc_output = cond_gan.disc_forward(gen_output, c)
        self.assertTrue(np.array_equal(disc_output.shape, (4, 1)))

    def test_conv_networks_with_fusion(self):

        cond_gan = nets.ConditionalConvGAN(
            num_z_units=10,
            num_cond_vals=2,
            image_dim=28,
            image_channels=1,
            g_num_filters=64,
            d_num_filters=8,
            use_fusion_layer=True)

        z = np.random.normal(size=(4, 10)).astype(np.float32)
        z = torch.tensor(z)

        c = np.random.uniform(size=(4, 2)).astype(np.float32)
        c = torch.tensor(c)

        gen_output = cond_gan.gen_forward(z, c)
        self.assertTrue(np.array_equal(gen_output.shape, (4, 1, 28, 28)))

        disc_output = cond_gan.disc_forward(gen_output, c)
        self.assertTrue(np.array_equal(disc_output.shape, (4, 1)))
