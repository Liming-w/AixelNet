# AixelNet
## Project Overview
This is the implementation code of the paper **AixelNet: A Pre-trained Model with Table-aware Adaptation for Structured Data Prediction**.

## Project Structure
This is the structure of the AixelNet project:

```latex
├── dataset/                       // Datasets for pre-training and fine-tuning
│   ├── finetune/                  // Fine-tuning example dataset
│   ├── pretrain/                  // Pre-training datasets
├── AixelNet/                        // Core implementation of AixelNet model
│   ├── tokenizer/                 // Tokenizer files for embedding
│   ├── constants.py               // Constant values used in the project
│   ├── evaluator.py               // Evaluation logic for model performance
│   ├── load_pretrain_data.py      // Pre-training dataset loading logic
│   ├── modeling_AixelNet.py         // AixelNet model definition (main model)
│   ├── AixelNet.py                  // AixelNet model logic and utilities
│   ├── regularization.py          // Regularization methods for model optimization
│   ├── trainer.py                 // Model training code
│   ├── trainer_utils.py           // Utility functions for training
│   ├── bo_optimizer.py            // Hyperparameter optimization using Bayesian Optimization
├── run_finetune.py                // Script for fine-tuning the AixelNet
├── run_pretrain.py                // Script for pre-training the AixelNet
└── requirements.txt               // Required dependencies for the project
```



## Requirements
+ scikit-learn==1.2.2
+ scipy==1.11.2
+ tokenizers==0.13.3
+ transformers==4.24.0
+ torch==1.12.1+cu116

## Datasets
The pre-training dataset is sourced from [OpenTabs](https://mega.nz/file/oqUlgbCa#AwNrJD6RDTIroZbJhMUIe5hS2y_DpBGMPLnsutZcAL0), a large-scale repository of structured tabular data curated for pre-training tasks. Please download and extract the dataset into the `dataset/pretrain/` directory.

The fine-tuning datasets are collected from multiple sources, including **Kaggle**, **UCI**, and **OpenML**, and should be placed under the `dataset/finetune/` directory.

Ensure that all required datasets are correctly prepared in their respective directories before running the experiments.  

## Run AixelNet model
### 1. **Download Pre-trained Datasets**
Download the pre-training datasets from [OpenTabs](https://mega.nz/file/oqUlgbCa#AwNrJD6RDTIroZbJhMUIe5hS2y_DpBGMPLnsutZcAL0) and extract them into `dataset/pretrain/`.

### 2. Build Pre-training Model
After downloading the datasets, run the pre-training script using the following command:

```bash
nohup bash -c "python -u run_pretrain.py --num_epoch 100 --num_data 2500 --lable_data_args dataset/pretrain/ --num_k_model 6 --save_model ./AixelNet --log_path ./logs/AixelNet.txt" &
```

### 3. Fine-tuning Model on Different Tables
Once the pre-training is complete, you can fine-tune the model with your dataset by running:

```bash
nohup bash -c "python -u finetune_example.py --cpt ./AixelNet-v0 --num_k_model 6 --model_name AixelNet --finetune_data_args dataset/finetune/cls/ --datasets VulNoneVul" &
```
