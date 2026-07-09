# LLM-Seg

A guideline-informed multimodal framework for text-guided 3D organ-at-risk segmentation in pancreatic SBRT.

---

## Introduction

**LLM-Seg** is a text-guided 3D segmentation framework designed to improve organ-at-risk (OAR) delineation for pancreatic cancer stereotactic body radiotherapy (SBRT). Unlike conventional image-only auto-segmentation models, LLM-Seg incorporates clinical contouring knowledge from consensus guidelines into a deep learning segmentation pipeline.

The framework uses large language models to extract organ-level contouring guidance from clinical guidelines, encodes the resulting text with BioBERT, and integrates the text features with CT image features in a Swin Transformer-based 3D segmentation network.

This repository provides the implementation of our guideline-informed multimodal segmentation framework for pancreatic SBRT OAR auto-contouring.

![LLM-Seg Overview](assets/Main_workflow.png)

---

## Framework Overview

LLM-Seg consists of three main stages:

### Stage 1: Guideline Extraction and Text Annotation

Clinical contouring guidance is extracted from consensus guidelines, including:

- ASTRO pancreas guideline
- ESTRO pancreas guideline
- ASTRO clinical practice guideline
- NRG Oncology International Consensus Contouring Atlas

GPT-4o is used to generate structured organ-level text annotation reports, followed by expert review.

### Stage 2: Text Encoding

The organ-level annotation reports are encoded using BioBERT. The resulting text embeddings are projected into a fixed-length feature representation using a multilayer perceptron (MLP).

### Stage 3: Text-Guided 3D Segmentation

The encoded text features are integrated with CT image features in a 3D segmentation network to guide OAR delineation. The framework is designed to improve segmentation robustness for anatomically complex structures such as the duodenum, stomach, bowel, liver, kidneys, and spinal cord.

---

## Features

- Guideline-informed 3D OAR segmentation for pancreatic SBRT
- Text-guided multimodal fusion of clinical knowledge and CT imaging
- GPT-4o-based organ-level guideline extraction
- BioBERT-based clinical text encoding
- Swin Transformer-based 3D segmentation backbone
- Evaluation on public and institutional pancreatic SBRT datasets
- Support for geometric and dosimetric evaluation

---

## Dataset

The framework was developed using CT images from the public TotalSegmentator dataset and externally validated on an institutional pancreatic SBRT cohort.

Due to data-sharing restrictions, institutional CT scans and clinical treatment plans are not publicly released.

### Public Dataset

Please download the public CT data directly from the official TotalSegmentator source:

```bash
https://github.com/wasserth/TotalSegmentator
