# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:22:39 2025
Prompt-guided LightningModule with per-class Dice evaluation
@author: leiy04
"""

import torch
import pytorch_lightning as pl
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from model import MedPromptSegNet


class MedPromptSegLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, num_classes=6, label_names=None):
        super().__init__()
        self.model = MedPromptSegNet(num_classes=num_classes)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.lr = lr
        self.num_classes = num_classes
        self.label_names = label_names if label_names else [f"class_{i}" for i in range(num_classes)]

    def forward(self, x, prompts):
        return self.model(x, prompts)

    def training_step(self, batch, batch_idx):
        img, mask, prompts = batch['image'], batch['mask'], batch['prompt_texts']
        logits = self(img, prompts)
        loss = self.loss_fn(logits, mask.unsqueeze(1))
        self.log('train_loss', loss, prog_bar=True)

        # Optional: compute train dice
        with torch.no_grad():
            mask_onehot = one_hot(mask.unsqueeze(1).long(), num_classes=self.num_classes)
            mask_onehot = mask_onehot.squeeze(1).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
            self.print(f"logits shape: {logits.shape}, mask_onehot: {mask_onehot.shape}")
            dice_scores = self.dice_metric(logits, mask_onehot)  # [C]
            
            for i, name in enumerate(self.label_names[1:], start=1):  # skip background
                self.log(f"train_dice_{name}", dice_scores[i - 1], prog_bar=False)
            self.log("train_mean_dice", dice_scores.mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, prompts = batch['image'], batch['mask'], batch['prompt_texts']
        logits = self(img, prompts)
        loss = self.loss_fn(logits, mask.unsqueeze(1))
        self.log('val_loss', loss, prog_bar=True)

        mask_onehot = one_hot(mask.unsqueeze(1).long(), num_classes=self.num_classes)
        mask_onehot = mask_onehot.squeeze(1).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        dice_scores = self.dice_metric(logits, mask_onehot)

        for i, name in enumerate(self.label_names[1:], start=1):  # skip background
            self.log(f"val_dice_{name}", dice_scores[i - 1], prog_bar=False)
        self.log("val_mean_dice", dice_scores.mean(), prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)