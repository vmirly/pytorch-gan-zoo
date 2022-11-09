import io
import os.path as osp
import sys
from setuptools import setup, find_packages


NAME = "pytorch-gan-zoo"
DESCRIPTION = "Implementation of GAN models in PyTorch."
URL = "https://github.com/vmirly/pytorch-gan-zoo"
AUTHOR = "Vahid Mirjalili"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.0.1"  # ganzoo.__version__


try:
    with io.open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

EXTRAS = {
    "test": [
        "pytest",
        "mock",
        "flake8==4.0.1",
        "flake8-docstrings==1.6.0"
    ]
}


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(include=['ganzoo', 'ganzoo.*'])
)
