import unittest
import torch
import numpy as np
from networks import basic_gan_networks as nets


class TestBasicGANNetworks(unittest.TestCase):

    def test_fc_networks(self):
        gen = nets.make_generator(10, 16, 28, 1, 0.5)
        disc = nets.make_discriminator(28*28, 16, 0.5)

        z = np.random.normal(size=(1, 10)).astype(np.float32)
        z = torch.tensor(z)

        gen_output = gen(z)
        self.assertTrue(np.array_equal(gen_output.shape, (1, 1, 28, 28)))

        disc_output = disc(gen_output)
        self.assertTrue(np.array_equal(disc_output.shape, (1, 1)))

    def test_conv_networks(self):
        gen = nets.make_conv_generator(10, 64, 28, 1)
        disc = nets.make_conv_discriminator(28, 1, 16)

        z = np.random.normal(size=(4, 10)).astype(np.float32)
        z = torch.tensor(z)

        gen_output = gen(z)
        self.assertTrue(np.array_equal(gen_output.shape, (4, 1, 28, 28)))

        disc_output = disc(gen_output)
        self.assertTrue(np.array_equal(disc_output.shape, (4, 1)))
