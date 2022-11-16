"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""

import os
from pathlib import Path
from typing import Tuple, Union 
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision import datasets

import pytorch_lightning as pl


root_dir = os.path.join(Path.home(), '.ganzoo_data')


class LitVisionDataset(pl.LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            transform: torchvision.transforms,
            splits: Union[Tuple[int], Tuple[float]],
            batch_size: int = 32):

        super().__init__()

        self.dataset_name = dataset_name
        self.transform = transform
        self.splits = splits
        self.batch_size = batch_size

    def prepare_data(self):
        if self.dataset_name == 'mnist':
            datasets.MNIST(root_dir, train=True, download=True)
        elif self.dataset_name == 'fashion-mnist':
            datasets.FashionMNIST(root_dir, train=True, download=True)
        elif self.dataset_name == 'celeba':
            datasets.CelebA(root_dir, split='all', download=True)
        elif self.dataset_name == 'cifar10':
            datasets.CIFAR10(root_dir, train=True, download=True)
        elif self.dataset_name == 'cifar100':
            datasets.CIFAR100(root_dir, train=True, download=True)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size)
        return val_loader

    def setup(self, stage):
        if self.dataset_name == 'mnist':
            ds = datasets.MNIST(
                root_dir, train=True, download=False,
                transform=self.transform)
        elif self.dataset_name == 'fashion-mnist':
            ds = datasets.FashionMNIST(
                root_dir, train=True, download=False,
                transform=self.transform)
        elif self.dataset_name == 'celeba':
            ds = datasets.CelebA(
                root_dir, split='all', download=False,
                transform=self.transform)
        elif self.dataset_name == 'cifar10':
            ds = datasets.CIFAR10(
                root_dir, train=True, download=False,
                transform=self.transform)
        elif self.dataset_name == 'cifar100':
            ds = datasets.CIFAR100(
                root_dir, train=True, download=False,
                transform=self.transform)

        self.train_data, self.val_data = random_split(ds, self.splits)
