# AixelNet

## Project Overview

This repository contains the implementation of **AixelNet: A Pre-trained Model with Table-aware Adaptation for Structured Data Prediction**.

## Project Structure

```text
├── dataset/                       # Datasets for pre-training and fine-tuning
│   ├── finetune/                  # Fine-tuning example datasets
│   └── pretrain/                  # Pre-training datasets
├── AixelNet/                      # Core implementation
│   ├── tokenizer/                 # Tokenizer files
│   ├── extractor/                 # Feature extractor files
│   ├── constants.py               # Constant values
│   ├── evaluator.py               # Evaluation utilities
│   ├── load_pretrain_data.py      # Dataset loading logic
│   ├── meta_features.py           # Meta-feature extraction
│   ├── modeling_AixelNet.py       # Model definitions
│   ├── AixelNet.py                # Build, train, evaluate, and predict APIs
│   ├── regularization.py          # Regularization modules
│   ├── trainer.py                 # Training loop
│   ├── trainer_utils.py           # Training utilities
│   └── bo_optimizer.py            # Meta-feature-based hyperparameter selection
├── AixelNet-v0/                   # Released pre-trained checkpoint
├── finetune_example.py            # Fine-tuning example
├── run_pretrain.py                # Pre-training entry point
└── requirements.txt               # Python dependencies
```

## Requirements

+ scikit-learn==1.2.2
+ scipy==1.11.2
+ tokenizers==0.13.3
+ transformers==4.24.0
+ torch==1.12.1+cu116

## Datasets

The pre-training data follows the OpenTabs format. Download it from [OpenTabs](https://mega.nz/file/oqUlgbCa#AwNrJD6RDTIroZbJhMUIe5hS2y_DpBGMPLnsutZcAL0) and place the extracted CSV files under `dataset/pretrain/`.

Fine-tuning datasets should be placed under `dataset/finetune/`. This release includes example classification and regression datasets under `dataset/finetune/`.

## Run AixelNet

### 1. Prepare Pre-training Data

Download the OpenTabs pre-training data from [OpenTabs](https://mega.nz/file/oqUlgbCa#AwNrJD6RDTIroZbJhMUIe5hS2y_DpBGMPLnsutZcAL0), then place the extracted CSV files under `dataset/pretrain/`.

### 2. Build a Pre-trained Model

```bash
python -u run_pretrain.py \
  --num_epoch 100 \
  --num_data 2500 \
  --label_data_args dataset/pretrain/ \
  --num_k_model 6 \
  --save_model ./AixelNet-pretrained \
  --log_path ./pretrain.log
```

### 3. Fine-tune on a Downstream Table

The released checkpoint is stored in `AixelNet-v0/`.

```bash
python -u finetune_example.py \
  --cpt ./AixelNet-v0 \
  --num_k_model 6 \
  --model_name AixelNet-v0 \
  --finetune_data_args dataset/finetune/cls/ \
  --datasets VulNoneVul
```
