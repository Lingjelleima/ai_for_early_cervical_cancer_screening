# Unified AI Framework for Early Cervical Cancer Screening

## Overview
This project implements a multi-source, anomaly-detection framework for early cervical cancer screening under unpaired data. The central challenge is that cytology images, colposcopy images, and clinical-history variables come from different sources and do not share patient identifiers, so a single fully supervised multimodal model cannot be trained or validated end-to-end.

To address this, the project builds three independent normative pipelines:

- Cytology anomaly detection
- Colposcopy anomaly detection
- Clinical-history / risk-factor anomaly detection

Each pipeline produces a robust anomaly score on a common normalized scale. These scores can later be fused in a missingness-aware late-fusion stage once a linked patient cohort becomes available. Importantly, no patient-level fusion is computed in the current version.

## Research Goal
To transition cervical cancer screening from fragmented, independent tests to a unified, explainable, and observability-driven framework that can combine macroscopic tissue patterns, microscopic cellular morphology, and molecular/clinical risk factors for early malignancy detection.

## Problem Setting
The notebook treats screening as a multi-modal anomaly detection problem rather than a conventional classification problem:

- Different modalities originate from different datasets and often lack aligned patient IDs
- Normal/reference cohorts are used to learn what is expected
- Abnormal findings are treated as deviations from the normative manifold
- Models are evaluated with AUROC, average precision (AP), thresholds, and group-aware bootstrap confidence intervals
- Group leakage is explicitly avoided by splitting at the patient or parent-image level

## Datasets Used
The framework is designed around multiple public and curated sources:

- Cytology datasets:
  - APCEM-Enhanced Cervical Cytology Image Dataset
  - Malhari Dataset
  - SIPaKMeD
- Colposcopy dataset:
  - DAI (Data for Artificial Intelligence-assisted colposcopy)
- Clinical risk factors dataset:
  - A structured cervical cancer risk-factor table used for independent history-based anomaly modeling

## Methodology
The implemented pipeline follows the research design described in `cervical_pipeline_v2.ipynb`:

### 1. Group-aware data preparation
- Images are collected across datasets and labeled as normal vs abnormal
- Splits are created by patient or parent-image group to prevent leakage across train/validation/test partitions
- The normal cohort is used for reference training and validation
- Abnormal samples are reserved for labeled validation and testing

### 2. Image preprocessing
- Images are shrunk once to 224x224 for efficient training and feature extraction
- Each dataset is assigned a modality (cytology or colposcopy)
- Source-specific metadata is preserved for evaluation, including image type (crop vs field) and group assignments

### 3. Frozen feature extraction
- A pretrained ResNet-50 acts as the teacher embedding model
- ViT-S and DINO-based student models are trained via distillation to match teacher embeddings on normal/reference data
- Distilled features represent the main learned image encoder used in the paper

### 4. Normative anomaly detection
For each modality, anomaly detection is performed using a set of normative models trained only on reference-normal data:

- Autoencoder reconstruction error
- Latent-space centroid distance
- Isolation Forest on latent features
- One-Class SVM on latent features
- Mahalanobis distance baseline
- k-NN baseline

Scores are normalized using a robust median/MAD calibration from validation-reference scores (Eq. 7), then combined using equal-weight aggregation rules (Eq. 8–9).

### 5. Evaluation and thresholding
The project evaluates:

- AUROC
- Average precision (AP)
- AP lift over prevalence
- Threshold-based operating points (sensitivity, specificity, precision, F1)
- Bootstrap confidence intervals with cluster-based resampling
- Paired statistical comparisons between scorers

### 6. Optional analyses
The notebook includes additional, clearly marked exploratory analyses beyond the paper's original v1 setup:

- Label-assisted scorer selection on the labeled validation split
- Repeated group-level re-splitting analysis

These analyses are intentionally separated from the core reported results.

## Core Pipeline Structure
The project is organized around a full research notebook workflow:

1. Environment setup and reproducibility logging
2. Dataset download and archive extraction
3. Manifest creation and group assignment
4. Data splitting by patient or parent-image groups
5. Image shrinking and caching
6. Teacher feature extraction
7. Student distillation (ResNet-50 -> ViT-S)
8. Autoencoder and detector training
9. Score normalization and aggregation
10. Benchmarking, thresholds, and bootstrap reporting

## Dependencies
The notebook uses a modern Python ML stack, including:

- Python 3.13
- PyTorch 2.11+
- timm
- scikit-learn
- pandas
- NumPy
- Pillow
- KaggleHub

## Repository Contents
- `README.md` – project overview and methodology
- `cervical_pipeline_v2.ipynb` – full research pipeline and experiments

## Intended Use
This repository is meant for research and exploratory screening-model development. It is not a production clinical deployment system. The current design supports normalization and cross-modality evaluation under incomplete or unpaired data, with a clear pathway to late-fusion once a linked patient cohort is available.

## Collaborators
- @Lingjelleima
- @nishita32
- @varshithakrishnappa
- @zuwairiya-lubna

## Tags
#medical-ai #healthcare #oncology
AI/ML: #unsupervised-learning #anomaly-detection #vit #transformers #xai
Tools: #pytorch #python #shap #isolation-forest #ocsvm
Status: #active-development #phase-1-research

## Notes
This repository reflects the versioned notebook approach used in the research workflow: the core methodology is reproducible, but the final late-fusion stage remains intentionally deferred until a matched patient-level cohort is available.
