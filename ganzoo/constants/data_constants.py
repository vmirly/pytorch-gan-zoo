"""
PyTorch GAN Zoo - data constants
Author: Vahid Mirjalili
"""
from ganzoo.constants import names

# Image dimensions
IMAGE_SHAPES = {
    names.NAMESTR_MNIST: (28, 28, 1),
    names.NAMESTR_FASHIONMNIST: (28, 28, 1),
    names.NAMESTR_CIFAR10: (32, 32, 3),
    names.NAMESTR_CIFAR100: (32, 32, 3),
    names.NAMESTR_CELEBA: (218, 178, 3),

    names.NAMESTR_IMAGEFOLDER: None 
}

NUM_CLASSES = {
    names.NAMESTR_MNIST: 10,
    names.NAMESTR_FASHIONMNIST: 10,
    names.NAMESTR_CIFAR10: 10,
    names.NAMESTR_CIFAR10: 100,
    names.NAMESTR_CELEBA: None,

    names.NAMESTR_IMAGEFOLDER: None
}
