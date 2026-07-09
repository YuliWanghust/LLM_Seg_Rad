# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:23:57 2025

@author: leiy04
"""

import pytorch_lightning as pl
from datamodule import DicomSegDataModule
from lightning_module import MedPromptSegLightning

expected_structs = {
    'GTV': ['GTV', 'GTVp'],
    'SmallBowel': ['SmallBowel'],
    'LargeBowel': ['LargeBowel'],
    'SpinalCord': ['SpinalCord'],
    'Liver': ['Liver'],
    'Duodenum': ['Duodenum'],
    'Stomach': ['Stomach'],
    'Kidney Lt': ['Kidney_L'],
    'Kidney Rt': ['Kidney_R']
}

prompt_map = {
    "GTV": "Segment the gross tumor volume of the pancreas based on visible contrast-enhanced CT abnormalities.",
    "Duodenum": "Segment the entire duodenum from the pylorus to the ligament of Treitz, with a 2–3 cm margin around the pancreatic tumor.",
    "Stomach": "Segment the stomach including the fundus, body, and antrum, especially areas within 2–3 cm proximity to the tumor.",
    "SmallBowel": "Segment all visible small bowel loops that fall within 3 cm of the pancreatic tumor boundary.",
    "LargeBowel": "Segment the colon and rectum segments within the 3 cm area surrounding the pancreas, especially transverse and descending colon.",
    "SpinalCord": "Segment the spinal cord starting from T10 through L3 for SBRT planning to minimize dose to critical neural structures.",
    "Liver": "Segment the entire liver volume, including lobes and vasculature, as an organ-at-risk during pancreatic SBRT.",
    "Kidney Lt": "Segment the left kidney in full including cortex and medulla for dose sparing during pancreatic SBRT.",
    "Kidney Rt": "Segment the right kidney in full including cortex and medulla for dose sparing during pancreatic SBRT."
}

# Create DataModule
datamodule = DicomSegDataModule(
    data_dir="./data",
    expected_structs=expected_structs,
    prompt_map=prompt_map,
    batch_size=1,
    val_split=0.2
)

# Add label names to match expected_structs for per-class Dice logging
label_names = ["background"] + list(expected_structs.keys())

# Instantiate model with label names
model = MedPromptSegLightning(
    lr=1e-4,
    num_classes=len(label_names),
    label_names=label_names
)

# Lightning Trainer
trainer = pl.Trainer(
    precision=16,
    devices=1,
    accelerator="gpu",
    max_epochs=400
)

trainer.fit(model, datamodule=datamodule)
