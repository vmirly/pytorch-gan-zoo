import unittest
import numpy as np
from PIL import Image
from torchvision import transforms

from ops import data_ops


class TestImageConversion(unittest.TestCase):

    def test_tensor2image_wihout_unnormalize(self):
        img = Image.open('data/1.jpg')
        print('Original size:', img.size)

        tsfm = transforms.ToTensor()
        t_array = tsfm(img)
        print('Array shape:', t_array.shape)

        img_converted = data_ops.convert_tensor2image(
            t_array, unnormalize=False)

        print(img_converted.size)
        print(type(img_converted))

        self.assertTrue(np.all(img_converted.size == img.size))
