import unittest
import numpy as np
from PIL import Image
from torchvision import transforms

from losses import img_losses


class TestLossFunctions(unittest.TestCase):

    def test_l1_loss(self):
        img = Image.open('misc/example.jpg')

        img_width, img_height = img.size
        print('Width: {}  Height: {}'.format(
            img_width, img_height))

        img_a = img.crop([0, 0, img_width//2, img_height])
        img_b = img.crop([img_width//2, 0, img_width, img_height])
        print('image_A', img_a.size, 'image_B:', img_b.size)

        tsfm = transforms.ToTensor()
        t_a = tsfm(img_a)
        t_b = tsfm(img_b)
        print('Array shape:', t_a.shape)

        print(t_a.dtype, t_a.numpy().dtype)

        self.assertEqual(img_losses.l1_lossfn(t_a, t_a), 0.0)
        self.assertAlmostEqual(
            img_losses.l1_lossfn(t_a, t_b).item(),
            np.mean(np.abs(t_a.numpy() - t_b.numpy())),
            places=7)
