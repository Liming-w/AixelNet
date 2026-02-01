import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from AixelNet.load_pretrain_data import load_single_data_all
from AixelNet.modeling_AixelNet import MetaFeatureExtractor
from AixelNet.bo_optimizer import optimize_hyperparameters
import AixelNet
import warnings

warnings.filterwarnings("ignore")
AixelNet.random_seed(42)

cal_device = "cuda"


def log_config(args):
    log_name = args.log_name
    exp_dir = f"search_{log_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    exp_log_dir = Path("logs") / exp_dir
    setattr(args, "exp_log_dir", exp_log_dir)

    exp_log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(exp_log_dir / "log.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def parse_args():
    parser = argparse.ArgumentParser(description="AixelNet-finetune (Classification Only, KFold First Fold)")
    parser.add_argument(
        "--wo_arg",
        type=str,
        choices=["wo_table_aware_finetuning"],
        default=None,
        help="Ablation Experiment",
    )
    parser.add_argument("--log_name", type=str, default="AixelNet_finetune_cls_kfold1", help="task name")
    parser.add_argument("--cpt", type=str, default="./AixelNet", help="pretrain model checkpoint path")
    parser.add_argument("--model_name", type=str, default="AixelNet-base", help="pretrain model name")
    parser.add_argument("--num_k_model", type=int, default=3, help="nums of predictors")
    parser.add_argument(
        "--finetune_data_args",
        type=str,
        default="dataset/finetune/",
        help="downstream dataset folder path for fine-tuning (contains .csv files)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="optional: specify downstream dataset names (csv). If None, use all csv in folder.",
    )
    return parser.parse_args()


def extract_meta_features(df: pd.DataFrame):
    meta_feature_extractor = MetaFeatureExtractor()
    return meta_feature_extractor.extract_meta_features(df)


_args = parse_args()
log_config(_args)

# KFold (5 splits) but only use the FIRST fold
kf = KFold(n_splits=5, random_state=42, shuffle=True)

all_res = {}
model_name = _args.model_name

data_dir = Path(_args.finetune_data_args)

# If user specifies datasets, only use those; else load all csv in directory
if _args.datasets:
    task_dataset = []
    for name in _args.datasets:
        name = name.strip()
        if not name.endswith(".csv"):
            name += ".csv"
        p = data_dir / name
        if not p.exists():
            available = sorted([x.name for x in data_dir.glob("*.csv")])
            raise FileNotFoundError(f"Dataset not found: {p}. Available csv: {available[:50]}")
        task_dataset.append(str(p))
else:
    task_dataset = sorted([str(p) for p in data_dir.glob("*.csv")])

for table_file_path in task_dataset:
    data_name = Path(table_file_path).name
    logging.info(f"Start========>{data_name}_DataSet==========>")
    # Load dataset
    X, y, cat_cols, num_cols, bin_cols = load_single_data_all(table_file_path)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Classification only: num_class from label cardinality
    num_class = int(len(y.value_counts()))

    # Keep column schema format expected by AixelNet
    cat_cols = [cat_cols]
    num_cols = [num_cols]
    bin_cols = [bin_cols]

    AixelNet.random_seed(42)
    trn_idx, val_idx = next(kf.split(X, y))

    # Use iloc to avoid pandas indexing ambiguity
    train_data = X.iloc[trn_idx]
    train_label = y.iloc[trn_idx]
    X_test = X.iloc[val_idx]
    y_test = y.iloc[val_idx]

    # NOTE: KFold is NOT stratified; to keep behavior consistent with your old code,
    # we still stratify the inner train/val split (classification).
    X_train, X_val, y_train, y_val = train_test_split(
        train_data,
        train_label,
        test_size=0.2,
        random_state=0,
        stratify=train_label,
        shuffle=True,
    )

    # Prepare DataFrame for meta feature extraction
    df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    df_val = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Build classifier
    model = AixelNet.build_classifier(
        checkpoint=_args.cpt,
        device=cal_device,
        num_class=num_class,
        k_models=_args.num_k_model,
    )
    eval_metric = "auc"
    eval_less_is_better = False

    model.update({"cat": cat_cols, "num": num_cols, "bin": bin_cols})

    meta_features = extract_meta_features(df_train)

    # Training args (BO or default)
    if not _args.wo_arg:
        optimal_hyperparameters = optimize_hyperparameters(meta_features)
        training_arguments = {
            "num_epoch": optimal_hyperparameters["num_epoch"],
            "batch_size": optimal_hyperparameters["batch_size"],
            "lr": optimal_hyperparameters["lr"],
            "eval_metric": eval_metric,
            "eval_less_is_better": eval_less_is_better,
            "regression_task": False,
            "output_dir": f"./models/checkpoint-finetune-{data_name[:-4]}-{model_name}",
            "patience": 30,
            "num_workers": 0,
            "device": cal_device,
            "flag": 1,
        }
    else:
        training_arguments = {
            "num_epoch": 200,
            "batch_size": 64,
            "lr": 1e-4,
            "eval_metric": eval_metric,
            "eval_less_is_better": eval_less_is_better,
            "regression_task": False,
            "output_dir": f"./models/checkpoint-finetune-{data_name[:-4]}-{model_name}",
            "patience": 30,
            "num_workers": 0,
            "device": cal_device,
            "flag": 1,
        }

    # Remove output_dir if exists
    if os.path.isdir(training_arguments["output_dir"]):
        shutil.rmtree(training_arguments["output_dir"])

    # Trainer inputs
    train_input = ((X_train, y_train, df_train), 0)
    val_input = ((X_val, y_val, df_val), 0)
    test_input = ((X_test, y_test, df_test), 0)

    trainer = AixelNet.train(model, train_input, val_input, data_weight=[True], **training_arguments)
    _ = trainer.train(test_input)

    # Predict + evaluate
    ypred = AixelNet.predict_new(model, X_test, df_test=df_test, regression_task=False)
    ans = AixelNet.evaluate(ypred, y_test, metric=eval_metric, num_class=num_class)

    all_res[data_name] = float(ans[0])
    logging.info(f"Test_Score===>{data_name}_DataSet==> {all_res[data_name]}")
result_df = pd.DataFrame(
    [{"dataset": k, "result": v} for k, v in all_res.items()],
    columns=["dataset", "result"],
)
res_path = str(_args.exp_log_dir / "res.csv")
result_df.to_csv(res_path, index=False)