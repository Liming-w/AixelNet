import argparse
import logging
import os
import AixelNet
from loguru import logger
import warnings
from AixelNet.load_pretrain_data import load_all_label_data_for_pretrain  # New supervised data loading method

# Disable wandb for logging
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='AixelNet-ensemble-supervised-pretrain-ds')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument("--lable_data_args", type=str, default="../AixelNet/dataset/pretrain", help="Pretrain data path")
    parser.add_argument("--save_model", type=str, default="./AixelNet_model", help="Path to save the model")
    parser.add_argument("--num_data", type=int, default=1000, help="Number of pretrain datasets to use")
    parser.add_argument("--log_path", type=str, default="./logs/AixelNet.txt", help="Log file path")

    # Model specific hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--wo_arg", type=str, choices=['wo_meta', 'wo_hypernetwork', 'wo_sparse', 'wo_regularization'], default=None, help="Ablation Experiment")
    parser.add_argument("--hyper_arg", type=str, choices=['None', 'Uniform', 'Heuristic'], default=None, help="Hypernetwork Experiment")
    parser.add_argument("--lambda1", type=float, default=1e-4, help="Regularization Strength 1")
    parser.add_argument("--lambda2", type=float, default=1e-4, help="Regularization Strength 2")
    parser.add_argument("--num_layer", type=int, default=3, help="Number of layers")
    parser.add_argument("--ffn_dim", type=int, default=256, help="Feed-forward network dimension")
    parser.add_argument("--num_attention_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="Hidden dropout probability")

    # Training hyperparameters
    parser.add_argument("--num_epoch", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_k_model", type=int, default=3, help="Number of ensemble models")

    args = parser.parse_args()
    return args


# Initialize distributed training
_args = parse_args()
dev = f'cuda:{_args.local_rank}'
logger.info(f'dev: {dev}')
cal_device = 'cuda'

if "OMPI_COMM_WORLD_RANK" in os.environ:
    my_rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
elif "RANK" in os.environ:
    my_rank = int(os.getenv("RANK"))
else:
    my_rank = 0

log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

logger_config = {
    "handlers": [
        {
            "sink": _args.log_path,
            "level": log_level,
            "colorize": True,
            "format": "[rank {extra[rank]}] [{time}] [{level}] {message}",
        },
    ],
    "extra": {"rank": my_rank},
}
logger.configure(**logger_config)

if _args.hyper_arg == 'None':
    _args.hyper_arg = None

# Load supervised labeled data
trainset, valset, cat_cols, num_cols, bin_cols, dataset_paths = load_all_label_data_for_pretrain(
    label_data_path=_args.lable_data_args,
    seed=42,
    limit=_args.num_data,
    core_size=5000
)

# Check if dataset paths are found
if not dataset_paths:
    raise ValueError("No valid dataset paths found, please check if 'lable_data_args' is correct.")

# Automatically infer the number of classes for each dataset
num_classes_list = []
for train in trainset:
    labels = train[0][1]  # Extract labels for each task
    num_classes = len(set(labels)) if labels is not None else 2  # Assume binary classification by default
    num_classes_list.append(num_classes)

# Build the ensemble supervised model
model = AixelNet.build_pretrain_model(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    num_classes_list=num_classes_list, 
    device=cal_device,
    dataset_paths=dataset_paths, 
    k_models=_args.num_k_model, 
    hidden_dim=_args.hidden_dim,
    num_layer=_args.num_layer,
    num_attention_head=_args.num_attention_head,
    hidden_dropout_prob=_args.hidden_dropout_prob,
    ffn_dim=_args.ffn_dim,
    vocab_freeze=True,
    wo_arg=_args.wo_arg,
    lambda1 = _args.lambda1,
    lambda2 = _args.lambda2,
)

# Define training arguments
training_arguments = {
    'num_epoch': _args.num_epoch,
    'batch_size': _args.batch_size,
    'lr': _args.lr,
    'eval_metric': 'val_loss',  # Validation loss used for evaluation
    'eval_less_is_better': True,
    'output_dir': _args.save_model,
    'patience': _args.patience,
    'warmup_steps': 1,
    'num_workers': 0,
}
logging.info(training_arguments)

# Ensure that trainset and valset are in (features, labels) format
trainer = AixelNet.train(
    model=model,
    trainset=trainset,  # (features, labels)
    valset=valset,  # (features, labels)
    data_weight=None,
    train_method='normal',
    cmd_args=_args,
    **training_arguments
)

# Start training
trainer.train()
