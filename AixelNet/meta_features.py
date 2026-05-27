import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


M_T_DIM = 18
M_F_DIM = 25
M_TF_DIM = 22
META_FEATURE_DIM = M_T_DIM + M_F_DIM + M_TF_DIM
META_FEATURE_GROUP_DIMS = {
    "target": M_T_DIM,
    "feature": M_F_DIM,
    "target_feature": M_TF_DIM,
}
META_FEATURE_SPLIT_INDICES = [0, M_T_DIM, M_T_DIM + M_F_DIM, META_FEATURE_DIM]


def get_meta_feature_split_indices(meta_dim: Optional[int] = None):
    if meta_dim is None or meta_dim == META_FEATURE_DIM:
        return list(META_FEATURE_SPLIT_INDICES)
    return [0, meta_dim // 3, 2 * meta_dim // 3, meta_dim]


class MetaFeatureExtractor:
    """Fast, fixed-size table meta-feature extractor for AixelNet."""

    def __init__(self, groups=None, random_state: int = 42, max_probe_samples: int = 2000):
        self.groups = groups
        self.random_state = random_state
        self.max_probe_samples = max_probe_samples
        self.meta_feature_dim = META_FEATURE_DIM

    def extract_meta_features(
        self,
        df: pd.DataFrame,
        task_type: str = "cls",
        categorical_columns: Optional[Sequence[str]] = None,
        numerical_columns: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
    ) -> torch.Tensor:
        m_t, m_f, m_tf = self.extract_meta_feature_groups(
            df=df,
            task_type=task_type,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            target=target,
        )
        values = np.concatenate([m_t, m_f, m_tf]).astype(np.float32)
        values = self._clean_vector(values, META_FEATURE_DIM)
        self.meta_feature_dim = len(values)
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    def extract_meta_feature_groups(
        self,
        df: pd.DataFrame,
        task_type: str = "cls",
        categorical_columns: Optional[Sequence[str]] = None,
        numerical_columns: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if df is None or df.empty:
            return (
                np.zeros(M_T_DIM, dtype=np.float32),
                np.zeros(M_F_DIM, dtype=np.float32),
                np.zeros(M_TF_DIM, dtype=np.float32),
            )

        df = df.copy()
        target_col = target or df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])

        categorical_columns, numerical_columns = self._resolve_feature_types(
            X, categorical_columns=categorical_columns, numerical_columns=numerical_columns
        )
        binary_columns = self._find_binary_columns(X)
        is_classification = self._is_classification(task_type)

        m_t = self._target_features(y, X.shape[1], is_classification)
        m_f = self._feature_features(X, categorical_columns, numerical_columns, binary_columns)
        m_tf = self._target_feature_features(X, y, categorical_columns, numerical_columns, is_classification)
        return (
            self._clean_vector(m_t, M_T_DIM),
            self._clean_vector(m_f, M_F_DIM),
            self._clean_vector(m_tf, M_TF_DIM),
        )

    def _is_classification(self, task_type: str) -> bool:
        return str(task_type).lower() in {"cls", "classification", "classify", "binclass", "multiclass"}

    def _resolve_feature_types(
        self,
        X: pd.DataFrame,
        categorical_columns: Optional[Sequence[str]],
        numerical_columns: Optional[Sequence[str]],
    ) -> Tuple[list, list]:
        all_cols = list(X.columns)
        if categorical_columns is not None or numerical_columns is not None:
            categorical = [c for c in self._normalize_column_names(categorical_columns) if c in all_cols]
            numerical = [c for c in self._normalize_column_names(numerical_columns) if c in all_cols]
            assigned = set(categorical) | set(numerical)
            for col in all_cols:
                if col in assigned:
                    continue
                if self._is_numeric_column(X[col]) and X[col].nunique(dropna=True) >= 15:
                    numerical.append(col)
                else:
                    categorical.append(col)
            return categorical, numerical

        categorical, numerical = [], []
        for col in all_cols:
            if self._is_numeric_column(X[col]) and X[col].nunique(dropna=True) >= 15:
                numerical.append(col)
            else:
                categorical.append(col)
        return categorical, numerical

    def _normalize_column_names(self, columns: Optional[Sequence[str]]) -> list:
        if columns is None:
            return []
        return [str(col).lower() for col in columns]

    def _is_numeric_column(self, series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series) or pd.to_numeric(series, errors="coerce").notna().mean() > 0.8

    def _find_binary_columns(self, X: pd.DataFrame) -> list:
        binary = []
        for col in X.columns:
            if X[col].dropna().nunique() <= 2:
                binary.append(col)
        return binary

    def _target_features(self, y: pd.Series, num_features: int, is_classification: bool) -> np.ndarray:
        n_rows = len(y)
        out = np.zeros(M_T_DIM, dtype=np.float32)
        out[0] = 1.0 if is_classification else 0.0
        out[1] = 0.0 if is_classification else 1.0
        out[2] = float(num_features)
        out[3] = float(n_rows)
        out[4] = math.log1p(max(num_features, 0))
        out[5] = math.log1p(max(n_rows, 0))

        if is_classification:
            counts = y.dropna().value_counts()
            probs = (counts / max(counts.sum(), 1)).to_numpy(dtype=np.float64)
            if probs.size:
                out[6] = float(len(probs))
                out[7:11] = self._summary_stats(probs)
                out[11] = float(-(probs * np.log(probs + 1e-12)).sum())
                out[12] = float(probs.max() / max(probs.min(), 1e-12))
            return out

        y_num = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=np.float64)
        if y_num.size:
            out[13] = float(np.max(y_num) - np.min(y_num))
            out[14] = float(np.std(y_num))
            out[15] = self._safe_skew(y_num)
            out[16] = self._safe_kurtosis(y_num)
            out[17] = 0.0  # Stable placeholder for future normality score.
        return out

    def _feature_features(
        self,
        X: pd.DataFrame,
        categorical_columns: Sequence[str],
        numerical_columns: Sequence[str],
        binary_columns: Sequence[str],
    ) -> np.ndarray:
        n_rows, n_features = X.shape
        total_entries = max(n_rows * n_features, 1)
        out = np.zeros(M_F_DIM, dtype=np.float32)

        missing_by_col = X.isna().sum(axis=0).to_numpy(dtype=np.float64) if n_features else np.array([])
        missing_by_row = X.isna().sum(axis=1).to_numpy(dtype=np.float64) if n_rows else np.array([])

        num_categorical = len(categorical_columns)
        num_numerical = len(numerical_columns)
        num_binary = len(binary_columns)
        out[0] = float(num_categorical)
        out[1] = self._safe_ratio(num_categorical, n_features)
        out[2] = float(num_numerical)
        out[3] = self._safe_ratio(num_numerical, n_features)
        out[4] = float(num_binary)
        out[5] = self._safe_ratio(num_binary, n_features)
        out[6] = float(np.count_nonzero(missing_by_col))
        out[7] = self._safe_ratio(out[6], n_features)
        out[8] = float(missing_by_col.sum())
        out[9] = self._safe_ratio(out[8], total_entries)
        out[10] = float(np.count_nonzero(missing_by_row))
        out[11] = self._safe_ratio(out[10], n_rows)

        skew_values, kurtosis_values, std_values, zero_ratios = [], [], [], []
        for col in numerical_columns:
            values = pd.to_numeric(X[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            std = float(np.std(values))
            std_values.append(std)
            zero_ratios.append(float(np.mean(values == 0)))
            if values.size >= 3 and std > 1e-12:
                skew_values.append(self._safe_skew(values))
                kurtosis_values.append(self._safe_kurtosis(values))

        out[12:16] = self._summary_stats(skew_values)
        out[16:20] = self._summary_stats(kurtosis_values)
        out[20] = self._safe_ratio(n_features, n_rows)
        out[21] = self._safe_ratio(n_rows, n_features)
        std_summary = self._summary_stats(std_values)
        out[22] = std_summary[0]
        out[23] = std_summary[1]
        out[24] = self._summary_stats(zero_ratios)[0]
        return out

    def _target_feature_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_columns: Sequence[str],
        numerical_columns: Sequence[str],
        is_classification: bool,
    ) -> np.ndarray:
        out = np.zeros(M_TF_DIM, dtype=np.float32)
        if X.empty or len(y) == 0:
            return out

        encoded_X, discrete_mask = self._encode_features(X, categorical_columns, numerical_columns)
        if encoded_X.shape[1] == 0:
            return out

        valid_rows = ~pd.isna(y).to_numpy()
        encoded_X = encoded_X[valid_rows]
        X_valid = X.reset_index(drop=True).loc[valid_rows].reset_index(drop=True)
        y_valid = y.reset_index(drop=True)[valid_rows]
        if encoded_X.shape[0] == 0:
            return out

        if is_classification:
            y_encoded = pd.factorize(y_valid)[0]
            out[0:4] = self._summary_stats(self._mutual_information(encoded_X, y_encoded, discrete_mask, True))
            out[4:8] = self._summary_stats(self._conditional_entropy(encoded_X, y_encoded, discrete_mask))
            out[12:17] = self._classification_probe(encoded_X, y_encoded)
        else:
            y_num = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(y_num)
            encoded_X = encoded_X[finite_mask]
            X_valid = X_valid.loc[finite_mask].reset_index(drop=True)
            y_num = y_num[finite_mask]
            if y_num.size == 0:
                return out
            out[0:4] = self._summary_stats(self._mutual_information(encoded_X, y_num, discrete_mask, False))
            out[8:12] = self._summary_stats(self._pearson_correlations(X_valid, y_num, numerical_columns))
            out[17:22] = self._regression_probe(encoded_X, y_num)
        return out

    def _encode_features(
        self,
        X: pd.DataFrame,
        categorical_columns: Sequence[str],
        numerical_columns: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        encoded_cols = []
        discrete_mask = []
        categorical_set = set(categorical_columns)
        numerical_set = set(numerical_columns)

        for col in X.columns:
            series = X[col]
            if col in numerical_set and col not in categorical_set:
                values = pd.to_numeric(series, errors="coerce")
                fill_value = values.median()
                if not np.isfinite(fill_value):
                    fill_value = 0.0
                encoded_cols.append(values.fillna(fill_value).to_numpy(dtype=np.float64))
                discrete_mask.append(False)
            else:
                values = series.astype("object").where(series.notna(), "__missing__")
                encoded_cols.append(pd.factorize(values)[0].astype(np.float64))
                discrete_mask.append(True)

        if not encoded_cols:
            return np.zeros((len(X), 0), dtype=np.float64), np.array([], dtype=bool)
        encoded = np.vstack(encoded_cols).T
        encoded = np.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        return encoded, np.array(discrete_mask, dtype=bool)

    def _mutual_information(self, X_values, y_values, discrete_mask, is_classification: bool) -> np.ndarray:
        if X_values.shape[0] < 3 or X_values.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        try:
            if is_classification:
                if len(np.unique(y_values)) < 2:
                    return np.zeros(0, dtype=np.float32)
                values = mutual_info_classif(
                    X_values,
                    y_values,
                    discrete_features=discrete_mask,
                    random_state=self.random_state,
                )
            else:
                if np.nanstd(y_values) <= 1e-12:
                    return np.zeros(0, dtype=np.float32)
                values = mutual_info_regression(
                    X_values,
                    y_values,
                    discrete_features=discrete_mask,
                    random_state=self.random_state,
                )
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        except Exception:
            return np.zeros(0, dtype=np.float32)

    def _conditional_entropy(self, X_values, y_values, discrete_mask) -> np.ndarray:
        if len(np.unique(y_values)) < 2:
            return np.zeros(0, dtype=np.float32)

        values = []
        for col_idx in range(X_values.shape[1]):
            feature_values = X_values[:, col_idx]
            if not discrete_mask[col_idx]:
                feature_values = self._bin_numeric(feature_values)
            values.append(self._conditional_entropy_one(feature_values, y_values))
        return np.asarray(values, dtype=np.float32)

    def _conditional_entropy_one(self, feature_values, y_values) -> float:
        total = len(y_values)
        if total == 0:
            return 0.0
        entropy = 0.0
        for value in np.unique(feature_values):
            mask = feature_values == value
            y_group = y_values[mask]
            if y_group.size == 0:
                continue
            probs = np.bincount(y_group.astype(int)) / y_group.size
            probs = probs[probs > 0]
            entropy += (y_group.size / total) * float(-(probs * np.log(probs + 1e-12)).sum())
        return entropy

    def _pearson_correlations(self, X: pd.DataFrame, y_values: np.ndarray, numerical_columns: Sequence[str]) -> np.ndarray:
        correlations = []
        if y_values.size < 2 or np.nanstd(y_values) <= 1e-12:
            return np.zeros(0, dtype=np.float32)

        for col in numerical_columns:
            if col not in X.columns:
                continue
            x_values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            if finite.sum() < 2:
                continue
            x_finite = x_values[finite]
            y_finite = y_values[finite]
            if np.std(x_finite) <= 1e-12 or np.std(y_finite) <= 1e-12:
                continue
            corr = np.corrcoef(x_finite, y_finite)[0, 1]
            if np.isfinite(corr):
                correlations.append(corr)
        return np.asarray(correlations, dtype=np.float32)

    def _classification_probe(self, X_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        out = np.zeros(5, dtype=np.float32)
        unique, counts = np.unique(y_values, return_counts=True)
        if unique.size == 0:
            return out
        out[0] = float(counts.max() / counts.sum())
        if unique.size < 2 or X_values.shape[0] < 4:
            return out

        X_sample, y_sample = self._sample_rows(X_values, y_values)
        min_class_count = np.min(np.bincount(y_sample.astype(int)))
        n_splits = int(min(3, min_class_count))
        if n_splits < 2:
            return self._classification_probe_holdout(X_sample, y_sample, out)

        log_acc, log_f1, tree_acc, tree_f1 = [], [], [], []
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in splitter.split(X_sample, y_sample):
            X_train, X_test = X_sample[train_idx], X_sample[test_idx]
            y_train, y_test = y_sample[train_idx], y_sample[test_idx]
            self._append_classification_scores(X_train, X_test, y_train, y_test, log_acc, log_f1, tree_acc, tree_f1)

        out[1] = self._mean_or_zero(log_acc)
        out[2] = self._mean_or_zero(log_f1)
        out[3] = self._mean_or_zero(tree_acc)
        out[4] = self._mean_or_zero(tree_f1)
        return out

    def _classification_probe_holdout(self, X_values, y_values, out):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_values,
                y_values,
                test_size=0.3,
                random_state=self.random_state,
                stratify=y_values if np.min(np.bincount(y_values.astype(int))) >= 2 else None,
            )
            log_acc, log_f1, tree_acc, tree_f1 = [], [], [], []
            self._append_classification_scores(X_train, X_test, y_train, y_test, log_acc, log_f1, tree_acc, tree_f1)
            out[1] = self._mean_or_zero(log_acc)
            out[2] = self._mean_or_zero(log_f1)
            out[3] = self._mean_or_zero(tree_acc)
            out[4] = self._mean_or_zero(tree_f1)
        except Exception:
            pass
        return out

    def _append_classification_scores(self, X_train, X_test, y_train, y_test, log_acc, log_f1, tree_acc, tree_f1):
        try:
            clf = LogisticRegression(max_iter=300, random_state=self.random_state)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            log_acc.append(accuracy_score(y_test, pred))
            log_f1.append(f1_score(y_test, pred, average="macro", zero_division=0))
        except Exception:
            pass

        try:
            tree = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
            tree.fit(X_train, y_train)
            pred = tree.predict(X_test)
            tree_acc.append(accuracy_score(y_test, pred))
            tree_f1.append(f1_score(y_test, pred, average="macro", zero_division=0))
        except Exception:
            pass

    def _regression_probe(self, X_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        out = np.zeros(5, dtype=np.float32)
        if y_values.size < 4:
            return out
        X_sample, y_sample = self._sample_rows(X_values, y_values)
        if np.std(y_sample) <= 1e-12:
            return out

        n_splits = min(3, len(y_sample))
        if n_splits < 2:
            return out

        baseline_rmse, ridge_rmse, ridge_r2, tree_rmse, tree_r2 = [], [], [], [], []
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in splitter.split(X_sample):
            X_train, X_test = X_sample[train_idx], X_sample[test_idx]
            y_train, y_test = y_sample[train_idx], y_sample[test_idx]
            baseline = np.full_like(y_test, fill_value=np.mean(y_train), dtype=np.float64)
            baseline_rmse.append(mean_squared_error(y_test, baseline, squared=False))

            try:
                ridge = Ridge()
                ridge.fit(X_train, y_train)
                pred = ridge.predict(X_test)
                ridge_rmse.append(mean_squared_error(y_test, pred, squared=False))
                ridge_r2.append(r2_score(y_test, pred))
            except Exception:
                pass

            try:
                tree = DecisionTreeRegressor(max_depth=3, random_state=self.random_state)
                tree.fit(X_train, y_train)
                pred = tree.predict(X_test)
                tree_rmse.append(mean_squared_error(y_test, pred, squared=False))
                tree_r2.append(r2_score(y_test, pred))
            except Exception:
                pass

        out[0] = self._mean_or_zero(baseline_rmse)
        out[1] = self._mean_or_zero(ridge_rmse)
        out[2] = self._mean_or_zero(ridge_r2)
        out[3] = self._mean_or_zero(tree_rmse)
        out[4] = self._mean_or_zero(tree_r2)
        return out

    def _sample_rows(self, X_values: np.ndarray, y_values: np.ndarray):
        if X_values.shape[0] <= self.max_probe_samples:
            return X_values, y_values
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X_values.shape[0], size=self.max_probe_samples, replace=False)
        return X_values[indices], y_values[indices]

    def _bin_numeric(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2 or np.nanstd(values[finite]) <= 1e-12:
            return np.zeros_like(values, dtype=np.int64)
        try:
            bins = np.unique(np.nanquantile(values[finite], [0.2, 0.4, 0.6, 0.8]))
            return np.digitize(values, bins, right=False).astype(np.int64)
        except Exception:
            return np.zeros_like(values, dtype=np.int64)

    def _summary_stats(self, values: Iterable[float]) -> np.ndarray:
        values = np.asarray(list(values), dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.asarray(
            [np.mean(values), np.std(values), np.max(values), np.min(values)],
            dtype=np.float32,
        )

    def _safe_skew(self, values: np.ndarray) -> float:
        try:
            result = float(skew(values, bias=False, nan_policy="omit"))
            return result if np.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def _safe_kurtosis(self, values: np.ndarray) -> float:
        try:
            result = float(kurtosis(values, bias=False, nan_policy="omit"))
            return result if np.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def _safe_ratio(self, numerator, denominator) -> float:
        denominator = float(denominator)
        if abs(denominator) <= 1e-12:
            return 0.0
        result = float(numerator) / denominator
        return result if np.isfinite(result) else 0.0

    def _mean_or_zero(self, values: Sequence[float]) -> float:
        if not values:
            return 0.0
        result = float(np.mean(values))
        return result if np.isfinite(result) else 0.0

    def _clean_vector(self, values: np.ndarray, expected_dim: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != expected_dim:
            fixed = np.zeros(expected_dim, dtype=np.float32)
            fixed[: min(values.size, expected_dim)] = values[:expected_dim]
            values = fixed
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
