import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

from AixelNet.load_pretrain_data import TablePreprocessor, build_table_meta_features, load_single_data_all
from AixelNet.bo_optimizer import default_hyperparameters
import AixelNet
import warnings

warnings.filterwarnings("ignore")
AixelNet.random_seed(42)

def log_config(args):
    log_name = args.log_name
    exp_dir = f"search_{log_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    exp_log_dir = Path(args.log_dir) / exp_dir
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
        help="Ablation setting",
    )
    parser.add_argument("--log_name", type=str, default="AixelNet_finetune_cls_kfold1", help="task name")
    parser.add_argument("--cpt", type=str, default="./AixelNet-v0", help="pretrain model checkpoint path")
    parser.add_argument("--model_name", type=str, default="AixelNet-v0", help="pretrain model name")
    parser.add_argument("--num_k_model", type=int, default=6, help="number of predictors")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--num_epoch", type=int, default=None, help="Override fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional dataset sample limit")
    parser.add_argument("--output_dir", type=str, default="./models", help="Checkpoint parent directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log parent directory")
    parser.add_argument(
        "--finetune_data_args",
        type=str,
        default="dataset/finetune/cls/",
        help="downstream dataset folder path for fine-tuning (contains .csv files)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="optional: specify downstream dataset names (csv). If None, use all csv in folder.",
    )
    return parser.parse_args()


def extract_meta_features(df: pd.DataFrame, cat_cols, num_cols):
    return build_table_meta_features(
        df,
        task_type="cls",
        categorical_columns=cat_cols,
        numerical_columns=num_cols,
    )


_args = parse_args()
cal_device = _args.device
if torch.device(cal_device).type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable for the requested device")
log_config(_args)

kf = KFold(n_splits=5, random_state=42, shuffle=True)

all_res = {}
model_name = _args.model_name

data_dir = Path(_args.finetune_data_args)

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

if not task_dataset:
    raise ValueError(f"No datasets found in {data_dir}")

for table_file_path in task_dataset:
    data_name = Path(table_file_path).name
    logging.info(f"Start dataset: {data_name}")

    X, y, _, _, _ = load_single_data_all(
        table_file_path,
        task_type="cls",
        preprocess=False,
    )
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if _args.max_samples is not None:
        if _args.max_samples < 10:
            raise ValueError("max_samples must be at least 10")
        if len(X) > _args.max_samples:
            X, _, y, _ = train_test_split(
                X, y, train_size=_args.max_samples, random_state=42, stratify=y,
            )
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

    AixelNet.random_seed(42)
    trn_idx, val_idx = next(kf.split(X, y))

    train_data = X.iloc[trn_idx]
    train_label = y.iloc[trn_idx]
    X_test = X.iloc[val_idx]
    y_test = y.iloc[val_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        train_data,
        train_label,
        test_size=0.2,
        random_state=0,
        stratify=train_label,
        shuffle=True,
    )

    preprocessor = TablePreprocessor(task_type="cls").fit(X_train, y_train)
    y_train = preprocessor.transform_target(y_train)
    y_val = preprocessor.transform_target(y_val)
    y_test = preprocessor.transform_target(y_test)
    num_class = len(preprocessor.target_encoder.classes_)
    meta_target = "__aixelnet_target__"
    while meta_target in X_train.columns:
        meta_target += "_"
    df_for_meta = pd.concat([
        X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(meta_target),
    ], axis=1)
    table_meta_features = extract_meta_features(
        df_for_meta, preprocessor.categorical_columns, preprocessor.numerical_columns,
    )
    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    cat_cols = [preprocessor.categorical_columns]
    num_cols = [preprocessor.numerical_columns]
    bin_cols = [preprocessor.binary_columns]

    model = AixelNet.build_classifier(
        checkpoint=_args.cpt,
        device=cal_device,
        num_class=num_class,
        k_models=_args.num_k_model,
    )
    eval_metric = "auc"
    eval_less_is_better = False

    model.update({"cat": cat_cols, "num": num_cols, "bin": bin_cols})
    model.preprocessors = {0: preprocessor}

    if not _args.wo_arg:
        optimal_hyperparameters = default_hyperparameters()
        training_arguments = {
            "num_epoch": optimal_hyperparameters["num_epoch"],
            "batch_size": optimal_hyperparameters["batch_size"],
            "lr": optimal_hyperparameters["lr"],
            "eval_metric": eval_metric,
            "eval_less_is_better": eval_less_is_better,
            "regression_task": False,
            "output_dir": f"./models/checkpoint-finetune-{data_name[:-4]}-{model_name}",
            "patience": optimal_hyperparameters["patience"],
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

    training_arguments["output_dir"] = str(
        Path(_args.output_dir) / f"checkpoint-finetune-{data_name[:-4]}-{model_name}"
    )
    for name in ("num_epoch", "batch_size", "lr", "patience"):
        override = getattr(_args, name)
        if override is not None:
            training_arguments[name] = override
    if Path(training_arguments["output_dir"]).exists():
        raise FileExistsError(f"Checkpoint directory already exists: {training_arguments['output_dir']}")

    train_input = ((X_train, y_train, table_meta_features), 0)
    val_input = ((X_val, y_val, table_meta_features), 0)
    test_input = ((X_test, y_test, table_meta_features), 0)

    trainer = AixelNet.train(model, train_input, val_input, data_weight=[True], **training_arguments)
    _ = trainer.train()

    ypred = AixelNet.predict_new(model, X_test, meta_features=table_meta_features, regression_task=False)
    ans = AixelNet.evaluate(ypred, y_test, metric=eval_metric, num_class=num_class)

    all_res[data_name] = float(ans[0])
    logging.info(f"Test score for {data_name}: {all_res[data_name]}")

mean_list = [all_res[k] for k in all_res]
result_df = pd.DataFrame(
    [{"dataset": k, "result": v} for k, v in all_res.items()],
    columns=["dataset", "result"],
)
res_path = str(_args.exp_log_dir / "res.csv")
result_df.to_csv(res_path, index=False)
logging.info(f"Mean score: {float(np.mean(mean_list))}")
