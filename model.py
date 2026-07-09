# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:32:05 2025

@author: leiy04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from transformers import CLIPTokenizer, CLIPTextModel

# Decoder with Automatic Pathway (AP) Module
class APDecoder(nn.Module):
    def __init__(self, num_classes=6, prompt_dim=96, feature_dim=12, num_paths=3):
        super().__init__()
        self.num_paths = num_paths
        self.prompt_dim = prompt_dim
        self.feature_dim = feature_dim

        # Define multiple decoder paths
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(feature_dim // 2, feature_dim // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_paths)
        ])

        # Routing weights from prompt embedding to choose path
        self.routing_fc = nn.Linear(prompt_dim, num_paths)

        # Final 1x1x1 convolution to project to number of classes
        self.final_conv = nn.Conv3d(feature_dim // 4, num_classes, kernel_size=1)

    def forward(self, features, prompt_embedding):
        """
        Args:
            features: [B, C, D, H, W] (C == feature_dim)
            prompt_embedding: [B, prompt_dim]
        Returns:
            out: segmentation logits [B, num_classes, D, H, W]
        """
        # Get routing weights and apply softmax
        routing_weights = F.softmax(self.routing_fc(prompt_embedding), dim=-1)  # [B, num_paths]

        # Process through each path
        path_outputs = [path(features) for path in self.paths]  # List of [B, C', D, H, W]

        # Stack and weight outputs: [B, num_paths, C', D, H, W]
        stacked = torch.stack(path_outputs, dim=1)  # [B, num_paths, C', D, H, W]

        # Expand routing weights: [B, num_paths, 1, 1, 1, 1]
        weights = routing_weights.view(-1, self.num_paths, 1, 1, 1, 1)

        # Weighted sum of paths
        fused = (stacked * weights).sum(dim=1)  # [B, C', D, H, W]

        # Final classification
        out = self.final_conv(fused)  # [B, num_classes, D, H, W]
        return out
    
class MedPromptSegNet(nn.Module):
    def __init__(self, img_size = (512,512,128), in_channels=1, num_classes=6, feature_size=12, prompt_dim=512):
        super().__init__()

        # Image Encoder: SwinUNETR (MONAI)
        self.image_encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1,  # dummy output, not used
            feature_size=feature_size,
            use_checkpoint=False
        )

        # Prompt Encoder: CLIP Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")

        # Decoder: Automatic Pathway Decoder
        self.decoder = APDecoder(num_classes=num_classes, prompt_dim=prompt_dim, feature_dim=feature_size)

    def encode_prompt(self, prompt_texts):
        """
        Accepts: str, list of str, or nested lists.
        Returns: Tensor [B, 512]
        """
        # Handle single string
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]
    
        # Flatten if nested
        elif isinstance(prompt_texts, list):
            flat = []
            for p in prompt_texts:
                if isinstance(p, str):
                    flat.append(p)
                elif isinstance(p, list):
                    flat.extend(p)
            prompt_texts = flat
    
        # Fallback if empty
        if not prompt_texts:
            prompt_texts = ["Segment relevant organs."]
    
        # Tokenize safely
        tokens = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
        with torch.no_grad():
            embeddings = self.text_encoder(**tokens).last_hidden_state.mean(dim=1)  # [B, D]
        return embeddings

    def forward(self, img, prompt_texts):
        """
        img: [B, 1, D, H, W]
        prompt_texts: list of str
        returns: logits [B, num_classes, D, H, W]
        """
        x = self.image_encoder.encoder1(img)  # Use SwinUNETR's initial encoder output
        prompt_embedding = self.encode_prompt(prompt_texts)
        out = self.decoder(x, prompt_embedding)
        return out
