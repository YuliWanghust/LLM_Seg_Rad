# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:22:42 2025

@author: leiy04
"""

import os
import torch
import numpy as np
from lightning_module import MedPromptSegLightning
from helper import read_CT, get_lst
from convert import load_patient, struct_to_mask
from monai.transforms import Resize, EnsureChannelFirst
import torch.nn.functional as F
import argparse

# Dice Coefficient Calculation
def dice_coefficient(pred, gt, smooth=1e-6):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return (2. * intersection + smooth) / (union + smooth)

def run_inference(model_path, dicom_dir, prompt_map, expected_structs, device="cuda"):
    # Load model
    model = MedPromptSegLightning.load_from_checkpoint(model_path, num_classes=len(expected_structs)+1)
    model = model.model  # Extract underlying MedPromptSegNet
    model.to(device).eval()

    # Load CT image
    img, _ = read_CT(dicom_dir)  # [H, W, D]
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W, D]

    results = {}
    dice_scores = {}

    for struct_key, prompt in prompt_map.items():
        print(f"Running: {struct_key}")
        with torch.no_grad():
            logits = model(img, [prompt])  # [1, C, D, H, W]
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).squeeze().cpu().numpy()  # [D, H, W]
            
            # Ground truth mask
            gt_mask = struct_to_mask(get_lst(dicom_dir), expected_structs[struct_key][0])  # [D, H, W]
            gt_mask = np.transpose(gt_mask, (1, 2, 0))  # [H, W, D]
            
            # Calculate Dice score
            dice = dice_coefficient(pred_label, gt_mask)
            dice_scores[struct_key] = dice

            results[struct_key] = pred_label

    return results, dice_scores  # Dictionary of {label_name: 3D mask}, {label_name: dice_score}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .ckpt or .pt model")
    parser.add_argument("--dicom", type=str, required=True, help="Path to DICOM patient directory")
    args = parser.parse_args()

    expected_structs = {
        "GTV": ["GTV", "GTVp"], 
        "SmallBowel": ["SmallBowel"],
        "LargeBowel": ["LargeBowel"],
        "SpinalCord": ["SpinalCord"],
        "Liver": ["Liver"],
        "Duodenum": ["Duodenum"],
        "Stomach": ["Stomach"],
        "Kidney Lt": ["Kidney_L"],
        "Kidney Rt": ["Kidney_R"]
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

    seg_results, dice_scores = run_inference(args.model, args.dicom, prompt_map, expected_structs)

    # Print Dice Scores for each organ
    for label, dice in dice_scores.items():
        print(f"Dice Score for {label}: {dice:.4f}")
    
    # Optionally save the results (segmentation masks) to disk or DICOM format
    for label, mask in seg_results.items():
        print(f"{label} mask shape:", mask.shape)
        # Here you could save the masks, e.g., using np.save or converting to DICOM again
