# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:19:05 2025

@author: leiy04
"""

from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from dataset_builder import DicomPromptSegDataset

class DicomSegDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, expected_structs, prompt_map, batch_size=1, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.expected_structs = expected_structs
        self.prompt_map = prompt_map
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        full_dataset = DicomPromptSegDataset(
            data_dir=self.data_dir,
            expected_structs=self.expected_structs,
            prompt_map=self.prompt_map
        )
        val_len = int(len(full_dataset) * self.val_split)
        train_len = len(full_dataset) - val_len
        self.train_set, self.val_set = random_split(full_dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=54)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=54)