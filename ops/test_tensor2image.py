import unittest
import numpy as np
from PIL import Image
from torchvision import transforms

from ops import data_ops


class TestImageConversion(unittest.TestCase):

    def test_tensor2image(self):
        img = Image.open('misc/example.jpg')
        print('Original size:', img.size)

        tsfm = transforms.ToTensor()
        t_array = tsfm(img)
        print('Array shape:', t_array.shape)

        img_converted = data_ops.convert_tensor2image(
            t_array, unnormalize=False)

        print(img_converted.size)
        print(type(img_converted))

        img.save('/tmp/image-original-1.png')
        img_converted.save('/tmp/image-converted-1.png')

        self.assertTrue(np.all(img_converted.size == img.size))

    def test_tensor2image_with_normalize(self):

        img = Image.open('misc/example.jpg')

        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5))
        ])

        t_array = tsfm(img)
        img_converted = data_ops.convert_tensor2image(
            t_array, unnormalize=True)

        img.save('/tmp/image-original-2.png')
        img_converted.save('/tmp/image-converted-2.png')

        self.assertTrue(np.all(img_converted.size == img.size))
