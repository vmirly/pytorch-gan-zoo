import numpy as np
import torchvision.transforms.functional as F


class RandomPairedHFlip(object):
    """
    """

    def __init__(self, prob):
        assert isinstance(prob, float)
        self.prob = prob

    def __call__(self, image_pair):
        r = np.random.uniform(0, 1.0, size=None)

        if r > (1.0 - self.prob):
            return F.hflip(image_pair[0]), F.hflip(image_pair[1])
        else:
            return image_pair
