import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


DEFAULT_HYPERPARAMETERS = {
    "lr": 1e-4,
    "batch_size": 64,
    "num_epoch": 200,
    "patience": 30,
}

SEARCH_SPACE = {
    "lr": [3e-5, 1e-4, 3e-4, 1e-3],
    "batch_size": [32, 64, 128, 256],
    "num_epoch": [50, 100, 200],
    "patience": [10, 20, 30],
}


def default_hyperparameters():
    return dict(DEFAULT_HYPERPARAMETERS)


def flatten_meta_features(meta_features):
    arr = np.asarray(meta_features, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(float)


def normalize_config(config):
    if isinstance(config, dict):
        merged = default_hyperparameters()
        merged.update(config)
    else:
        values = list(config)
        if len(values) < 3:
            raise ValueError(f"Invalid hyperparameter config: {config}")
        merged = {
            "lr": values[0],
            "batch_size": values[1],
            "num_epoch": values[2],
            "patience": values[3] if len(values) > 3 else DEFAULT_HYPERPARAMETERS["patience"],
        }
    return {
        "lr": float(merged["lr"]),
        "batch_size": int(merged["batch_size"]),
        "num_epoch": int(merged["num_epoch"]),
        "patience": int(merged["patience"]),
    }


def build_candidate_configs(search_space=SEARCH_SPACE):
    keys = ["lr", "batch_size", "num_epoch", "patience"]
    return [normalize_config(dict(zip(keys, values))) for values in product(*[search_space[key] for key in keys])]


def _config_to_features(config):
    cfg = normalize_config(config)
    return np.array([
        math.log10(cfg["lr"]),
        math.log2(cfg["batch_size"]),
        cfg["num_epoch"] / 100.0,
        cfg["patience"] / 10.0,
    ], dtype=float)


def _objective_from_score(score, less_is_better):
    score = float(score)
    return -score if less_is_better else score


def load_history_data(history_path):
    path = Path(history_path)
    if not path.exists():
        return []

    text = path.read_text().strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(loaded, list):
        return loaded
    return [loaded]


def _record_objective(record, less_is_better=None):
    if "objective" in record and record["objective"] is not None:
        return float(record["objective"])
    if "score" in record and record["score"] is not None:
        direction = bool(record.get("less_is_better", less_is_better if less_is_better is not None else False))
        return _objective_from_score(record["score"], direction)
    if "loss" in record and record["loss"] is not None:
        return -float(record["loss"])
    return None


def _valid_history_records(history_data, task_type=None, metric=None, less_is_better=None):
    records = []
    for record in history_data:
        try:
            if task_type is not None and record.get("task_type") not in (None, task_type):
                continue
            if metric is not None and record.get("metric") not in (None, metric):
                continue
            if record.get("status", "ok") != "ok":
                continue

            meta = flatten_meta_features(record["meta_features"])
            config = normalize_config(record["config"])
            objective = _record_objective(record, less_is_better=less_is_better)
            if objective is None or not np.isfinite(objective):
                continue

            normalized = dict(record)
            normalized["meta_features"] = meta
            normalized["config"] = config
            normalized["objective"] = float(objective)
            records.append(normalized)
        except (KeyError, TypeError, ValueError):
            continue
    return records


def _filter_meta_dim(records, current_meta):
    return [record for record in records if len(record["meta_features"]) == len(current_meta)]


def _best_config(records):
    if not records:
        return default_hyperparameters()
    best = max(records, key=lambda record: record["objective"])
    return normalize_config(best["config"])


def _exact_meta_match_config(records, current_meta, atol=1e-8):
    matches = [
        record for record in records
        if len(record["meta_features"]) == len(current_meta)
        and np.allclose(record["meta_features"], current_meta, rtol=0.0, atol=atol)
    ]
    if not matches:
        return None
    return _best_config(matches)


def _fit_gp(X, y, random_state=42):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def _select_with_gp(records, current_meta, random_state=42):
    records = _filter_meta_dim(records, current_meta)
    if not records:
        return default_hyperparameters()

    X = np.vstack([
        np.concatenate([record["meta_features"], _config_to_features(record["config"])])
        for record in records
    ])
    y = np.asarray([record["objective"] for record in records], dtype=float)
    if float(np.std(y)) == 0.0:
        return _best_config(records)

    gp = _fit_gp(X, y, random_state=random_state)
    candidates = build_candidate_configs()
    candidate_X = np.vstack([
        np.concatenate([current_meta, _config_to_features(config)])
        for config in candidates
    ])
    mu, sigma = gp.predict(candidate_X, return_std=True)
    best_y = float(np.max(y))
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best_y) / sigma
    ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
    if not np.any(ei > 0):
        return normalize_config(candidates[int(np.argmax(mu))])
    return normalize_config(candidates[int(np.argmax(ei))])


def optimize_hyperparameters(
    meta_features,
    history_path="logs/bo_history.jsonl",
    min_history=8,
    task_type=None,
    metric=None,
    less_is_better=None,
    random_state=42,
):
    current_meta = flatten_meta_features(meta_features)
    history_data = load_history_data(history_path)
    records = _valid_history_records(
        history_data,
        task_type=task_type,
        metric=metric,
        less_is_better=less_is_better,
    )

    exact_config = _exact_meta_match_config(records, current_meta)
    if exact_config is not None:
        return exact_config

    records = _filter_meta_dim(records, current_meta)
    if len(records) < min_history:
        return default_hyperparameters()

    try:
        return _select_with_gp(records, current_meta, random_state=random_state)
    except Exception:
        return _best_config(records)
