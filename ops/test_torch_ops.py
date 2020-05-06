import torch
import unittest
import numpy as np

from ops import torch_ops


class TestTorchOps(unittest.TestCase):

    def test_onehot_embedding(self):

        a = torch.LongTensor([1, 0, 2, 1, 3, 2, 0, 3])

        onehot_encoder = torch_ops.onehot_embedding(4, device=None)

        a_hot = onehot_encoder(a)
        self.assertTrue(np.array_equal(a_hot.shape, (8, 4)))

    def test_flatten(self):

        x = torch.zeros((8, 4, 1, 3))

        f = torch_ops.Flatten()

        x_flat = f(x)
        self.assertTrue(np.array_equal(x_flat.shape, (8, 12)))

    def test_reshape(self):

        x = torch.zeros((8, 4, 1, 3))

        r1 = torch_ops.Reshape(new_shape=(12,))
        x_reshaped = r1(x)
        self.assertTrue(np.array_equal(x_reshaped.shape, (8, 12)))

        r2 = torch_ops.Reshape(new_shape=(2, 6))
        x_reshaped = r2(x)
        self.assertTrue(np.array_equal(x_reshaped.shape, (8, 2, 6)))
