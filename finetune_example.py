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

from AixelNet.load_pretrain_data import build_table_meta_features, load_single_data_all
from AixelNet.bo_optimizer import default_hyperparameters
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
        help="Ablation setting",
    )
    parser.add_argument("--log_name", type=str, default="AixelNet_finetune_cls_kfold1", help="task name")
    parser.add_argument("--cpt", type=str, default="./AixelNet-v0", help="pretrain model checkpoint path")
    parser.add_argument("--model_name", type=str, default="AixelNet-v0", help="pretrain model name")
    parser.add_argument("--num_k_model", type=int, default=6, help="number of predictors")
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

for table_file_path in task_dataset:
    data_name = Path(table_file_path).name
    logging.info(f"Start dataset: {data_name}")

    X, y, cat_cols, num_cols, bin_cols, df_for_meta = load_single_data_all(
        table_file_path,
        return_meta_frame=True,
        task_type="cls",
    )
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    num_class = int(len(y.value_counts()))

    table_meta_features = extract_meta_features(df_for_meta, cat_cols, num_cols)

    cat_cols = [cat_cols]
    num_cols = [num_cols]
    bin_cols = [bin_cols]

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

    model = AixelNet.build_classifier(
        checkpoint=_args.cpt,
        device=cal_device,
        num_class=num_class,
        k_models=_args.num_k_model,
    )
    eval_metric = "auc"
    eval_less_is_better = False

    model.update({"cat": cat_cols, "num": num_cols, "bin": bin_cols})

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

    if os.path.isdir(training_arguments["output_dir"]):
        shutil.rmtree(training_arguments["output_dir"])

    train_input = ((X_train, y_train, table_meta_features), 0)
    val_input = ((X_val, y_val, table_meta_features), 0)
    test_input = ((X_test, y_test, table_meta_features), 0)

    trainer = AixelNet.train(model, train_input, val_input, data_weight=[True], **training_arguments)
    _ = trainer.train(test_input)

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
