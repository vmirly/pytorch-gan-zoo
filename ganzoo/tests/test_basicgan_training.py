"""
PyTorch GAN Zoo
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from ganzoo.lit_modules import basic_fc_gan
from ganzoo.lit_modules import lit_data_custom
from ganzoo.constants import names


def test_lit_train_basicgan():

    arr_x = np.random.uniform(size=(4, 1, 7, 7)).astype('float32')
    arr_y = np.arange(len(arr_x)).astype('float32')
    ds = TensorDataset(torch.tensor(arr_x), torch.tensor(arr_y))
    # dl = DataLoader(ds, batch_size=2)
    dm = lit_data_custom.LitCustomDataset(ds, None, (0.5, 0.5), 2)

    for nt in names.LIST_FC_NETWORKS:
        lit_model = basic_fc_gan.LitBasicGANFC(
            num_z_units=4,
            z_distribution=names.NAMESTR_UNIFORM,
            num_hidden_units=8,
            image_dim=arr_x.shape[2],
            image_channels=arr_x.shape[1],
            p_drop=0.1,
            lr=0.1,
            beta1=0.5,
            beta2=0.9,
            network_type=nt)

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(lit_model, dm)
