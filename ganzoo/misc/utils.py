"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""

from typing import Tuple, Dict
from ganzoo.constants import names
from ganzoo.constants import defaults
from ganzoo.constants import data_constants as data_const


def get_dataset_props(dataset_name: str) -> Tuple[Tuple, str]:
    """
    Function to extract image shapes for a given dataset_name
    Arguments:
     - dataset_name: str

    Return:
     - image_shape: Tuple, shape of images in the dataset
     - msg: str, error message
    """

    if not dataset_name in data_const.IMAGE_SHAPES:
        msg = f'No info found for dataset {dataset_name}!'
        return None, msg

    image_shape = data_const.IMAGE_SHAPES[dataset_name]
    if not image_shape:
        return None, ''

    return image_shape, ''
