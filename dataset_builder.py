# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:42:17 2025

Updated dataset_builder.py with safe alias normalization
Target input shape: (512, 512, 128)

@author: leiy04
"""

import torch
import numpy as np
from monai.transforms import Resize
from torch.utils.data import Dataset
from helper import read_CT, get_lst
from convert import load_patient, struct_to_mask
from helper import get_struct_name, get_subdir

class DicomPromptSegDataset(Dataset):
    def __init__(self, data_dir, expected_structs, prompt_map, transform=None):
        """
        data_dir: path to DICOM patient folders
        expected_structs: dict like {"Duodenum": ["Duodenum"], ...}
        prompt_map: dict mapping OAR name to prompt text
        transform: optional transform to apply on (image, mask)
        """
        self.patient_dirs = sorted(get_subdir(data_dir))
        self.expected_structs = expected_structs
        self.prompt_map = prompt_map
        self.transform = transform

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        pth = self.patient_dirs[idx]
        dicom_files = get_lst(pth)
        patient = load_patient(dicom_files)
        img, _ = read_CT(pth)  # [H, W, D]
        print('CT shape:', img.shape)

        lst_structure_name = get_struct_name(patient)
        print(f"Patient {idx}: found structures: {lst_structure_name}")
        matched_structs = {}

        for std_name, aliases in self.expected_structs.items():
            for s in lst_structure_name:
                if any(alias.lower() == s.lower() for alias in aliases):
                    matched_structs[std_name] = s
                    break

        # Build mask and prompt list
        mask = np.zeros(img.shape, dtype=np.uint8)
        prompts = []
        for i, (k, v) in enumerate(matched_structs.items()):
            gt = struct_to_mask(dicom_files, v)  # [D, H, W]
            gt = np.transpose(gt, (1, 2, 0))      # [H, W, D]
            mask[gt > 0] = i + 1  # label values start at 1
            prompts.append(self.prompt_map.get(k, f"Segment {k}"))
        print('mask shape:', mask.shape)

        # Resize both image and mask to (512, 512, 128)
        #resize_img = Resize(spatial_size=(512, 512, 128), mode="bilinear")
        resize_img = Resize(spatial_size=(512, 512, 128), mode="trilinear", anti_aliasing=True)
        resize_mask = Resize(spatial_size=(512, 512, 128), mode="nearest")

        img = resize_img(img[np.newaxis])[0]     # [H, W, D]
        print(f"Resized image shape: {img.shape}")
        mask = resize_mask(mask[np.newaxis])[0]  # [H, W, D]
        print(f"Resized mask shape: {mask.shape}")
        # # ---- convert to channel-first 3D conv format: [C, D, H, W] ----
        img = np.transpose(img, (2, 1, 0))           # [D, H, W]
        mask = np.transpose(mask, (2, 1, 0))         # [D, H, W]
        # #img = img[np.newaxis, ...]                   # [1, D, H, W]

        sample = {
            'image': img.astype(np.float32),
            'mask': mask.astype(np.int64),
            'prompt_texts': prompts
        }

        if self.transform:
            sample = self.transform(sample)

        sample['image'] = torch.tensor(sample['image']).unsqueeze(0)  # [1, H, W, D]
        sample['mask'] = torch.tensor(sample['mask']).long()
        return sample