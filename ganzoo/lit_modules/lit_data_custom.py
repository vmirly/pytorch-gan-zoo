"""
PyTorch GAN Zoo
Author: Vahid Mirjalili
"""

import os
from typing import Tuple, Union 
import torch
from torch.utils.data import random_split, DataLoader
import torchvision

import pytorch_lightning as pl


class LitCustomDataset(pl.LightningDataModule):
    def __init__(
            self,
            torch_dataset: torch.utils.data.Dataset,
            transform: torchvision.transforms,
            splits: Union[Tuple[int], Tuple[float]],
            batch_size: int = 32):

        super().__init__()

        self.torch_dataset = torch_dataset
        self.transform = transform
        self.splits = splits
        self.batch_size = batch_size

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size)
        return val_loader

    def setup(self, stage):
        self.train_data, self.val_data = random_split(
            self.torch_dataset, self.splits
        )
