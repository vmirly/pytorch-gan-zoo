import unittest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from networks import pix2pix_networks as p2p_nets


class TestPix2PixNetworks(unittest.TestCase):

    @staticmethod
    def load_image():
        img = Image.open('./data/1.jpg')
        tsfm = transforms.ToTensor()
        example = tsfm(img)
        example_a = example[:, :, :256]
        example_b = example[:, :, 256:]

        return (torch.unsqueeze(example_a, axis=0),
                torch.unsqueeze(example_b, axis=0))

    def test_generator(self):
        generator = p2p_nets.Pix2PixGenerator()

        example_a, example_b = TestPix2PixNetworks.load_image()

        output_a = generator(example_a)
        self.assertTrue(np.array_equal(output_a.shape, example_a.shape))

        output_b = generator(example_b)
        self.assertTrue(np.array_equal(output_b.shape, example_b.shape))

    def test_discriminator(self):
        discriminator = p2p_nets.Pix2PixDiscriminator()

        example_a, example_b = TestPix2PixNetworks.load_image()

        input_real = torch.cat([example_a, example_b], axis=1)
        print(input_real.shape)

        d_real = discriminator(input_real)
        print(d_real.shape)

        generator = p2p_nets.Pix2PixGenerator()

        output_a = generator(example_a)

        input_fake = torch.cat([example_a, output_a], axis=1)
        print(input_fake.shape)


