import logging
import os
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .meta_features import MetaFeatureExtractor


logger = logging.getLogger(__name__)


def build_table_meta_features(df, task_type='cls', categorical_columns=None, numerical_columns=None):
    """
    Precompute table-level meta features once and pass the tensor through the
    training pipeline instead of carrying the full DataFrame in each batch.
    The last column is treated as the target unless an explicit target is
    passed to MetaFeatureExtractor later.
    """
    meta_feature_extractor = MetaFeatureExtractor()
    return meta_feature_extractor.extract_meta_features(
        df,
        task_type=task_type,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

class Feature_type_recognition():
    def __init__(self):
        self.df = None
    
    def get_data_type(self, col):
        # Determine the data type of the column (numeric or categorical).
        if 'std' in self.df[col].describe():
            if self.df[col].nunique() < 15:
                return 'cat'
            return 'num'
        else:
            return 'cat'

    def fit(self, df):
        # Fit the feature type recognition to the provided dataframe.
        self.df = df.infer_objects()
        self.num = []
        self.cat = []
        self.bin = []
        for col in self.df.columns:
            cur_type = self.get_data_type(col)
            if (cur_type == 'num'):
                self.num.append(col)
            elif (cur_type == 'cat'):
                self.cat.append(col)
            else:
                raise RuntimeError('Error in feature type!')
        return self.cat, self.bin, self.num
    
    def check_class(self, data_path):
        # Check if the target column is categorical or not.
        self.df = pd.read_csv(data_path)
        
        target_type = self.get_data_type(self.df.columns.tolist()[-1])
        if target_type == 'cat':
            return True
        else:
            return False

class TablePreprocessor:
    def __init__(self, task_type='cls', encode_cat=False, auto_feature_type=None, scale_target=False):
        self.task_type = task_type
        self.encode_cat = encode_cat
        self.auto_feature_type = copy.deepcopy(auto_feature_type) if auto_feature_type is not None else Feature_type_recognition()
        self.scale_target = scale_target
        self.feature_columns = None
        self.output_columns = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.binary_columns = []
        self.fill_values = {}
        self.feature_scaler = None
        self.categorical_encoder = None
        self.target_encoder = None
        self.target_scaler = None

    def _normalize(self, X):
        X = X.copy()
        X.columns = [str(col).lower() for col in X.columns]
        if X.columns.duplicated().any():
            raise ValueError('Feature names must be unique after normalization.')
        return X.replace([np.inf, -np.inf], np.nan)

    def _prepare(self, X):
        X = self._normalize(X)
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f'Missing feature columns: {sorted(missing)}')
        X = X.loc[:, self.feature_columns].copy()
        for col in self.numerical_columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(self.fill_values[col])
        for col in self.categorical_columns + self.binary_columns:
            values = X[col].astype('object')
            values = values.where(values.notna(), self.fill_values[col])
            X[col] = values.astype(str).str.lower()
        return X

    def fit(self, X_train, y_train=None):
        X = self._normalize(X_train)
        if X.empty:
            raise ValueError('Training features must contain rows and columns.')
        self.feature_columns = list(X.columns)
        type_frame = X.copy()
        for col in type_frame.columns:
            values = pd.to_numeric(type_frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            present = type_frame[col].notna()
            if present.any() and values[present].notna().mean() >= 0.8:
                type_frame[col] = values
        cat_cols, bin_cols, num_cols = self.auto_feature_type.fit(type_frame)
        self.categorical_columns = list(cat_cols)
        self.binary_columns = list(bin_cols)
        self.numerical_columns = list(num_cols)
        self.output_columns = self.binary_columns + self.numerical_columns + self.categorical_columns
        if len(self.output_columns) != len(self.feature_columns) or set(self.output_columns) != set(self.feature_columns):
            raise ValueError('Feature type recognition must assign every training column exactly once.')
        self.fill_values = {}
        for col in self.feature_columns:
            if col in self.numerical_columns:
                values = pd.to_numeric(X[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
                mode = values.mode(dropna=True)
                self.fill_values[col] = float(mode.iloc[0]) if not mode.empty else 0.0
            else:
                values = X[col].dropna().astype(str).str.lower()
                mode = values.mode(dropna=True)
                self.fill_values[col] = str(mode.iloc[0]) if not mode.empty else '__missing__'
        prepared = self._prepare(X)
        self.feature_scaler = None
        if self.numerical_columns:
            self.feature_scaler = MinMaxScaler().fit(prepared[self.numerical_columns])
        self.categorical_encoder = None
        if self.encode_cat and self.categorical_columns:
            self.categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(prepared[self.categorical_columns])
        self.target_encoder = None
        self.target_scaler = None
        if y_train is not None:
            if len(y_train) != len(X):
                raise ValueError('Training features and targets must have the same length.')
            if self.task_type == 'reg':
                values = pd.to_numeric(pd.Series(y_train), errors='coerce').to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError('Regression targets must be finite.')
                if self.scale_target:
                    self.target_scaler = StandardScaler().fit(values.reshape(-1, 1))
            else:
                self.target_encoder = LabelEncoder().fit(np.asarray(y_train))
        return self

    def transform(self, X):
        if self.feature_columns is None:
            raise RuntimeError('The preprocessor has not been fitted.')
        X = self._prepare(X)
        if self.numerical_columns:
            X[self.numerical_columns] = self.feature_scaler.transform(X[self.numerical_columns])
        if self.categorical_encoder is not None:
            X[self.categorical_columns] = self.categorical_encoder.transform(X[self.categorical_columns])
        for col in self.binary_columns:
            X[col] = X[col].isin(['yes', 'true', '1', 't']).astype(int)
        return X.loc[:, self.output_columns]

    def transform_target(self, y):
        if y is None:
            return None
        series = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
        if self.task_type == 'reg':
            values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Regression targets must be finite.')
            if self.target_scaler is not None:
                values = self.target_scaler.transform(values.reshape(-1, 1)).ravel()
        else:
            if self.target_encoder is None:
                raise RuntimeError('The target encoder has not been fitted.')
            values = self.target_encoder.transform(series.to_numpy())
        return pd.Series(values, index=series.index, name=series.name)

    def inverse_transform_target(self, y):
        values = np.asarray(y)
        if self.task_type == 'reg':
            if self.target_scaler is None:
                return values.copy()
            return self.target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)
        if self.target_encoder is None:
            raise RuntimeError('The target encoder has not been fitted.')
        return self.target_encoder.inverse_transform(values.astype(int).ravel()).reshape(values.shape)

def check_col_name_meaning(table_file, target, threshold=2):
    # Check if column names have meaningful or sufficient length.
    df = pd.read_csv(table_file)
    col_names = df.columns.values.tolist()
    col_names.remove(target)
    res = False
    good_cnt = 0
    one_cnt = 0
    for name in col_names:
        if len(name) <= 1:
            one_cnt += 1
            if one_cnt * 2 >= len(col_names):
                return False
        if not name[-1].isdigit():
            good_cnt += 1
            if good_cnt > threshold or good_cnt == len(col_names):
                res = True
    return res

def get_col_type(col):
    # Determine the type of the column (numeric or categorical).
    if 'std' in col.describe():
        if col.nunique() < 15:
            return 'cat'
        return 'num'
    else:
        return 'cat'

def check_word_count(text):
    # Check if the word count of the text is greater than or equal to 30.
    words = str(text).split()
    return len(words) >= 30

def check_data_quality(df):
    # Check the quality of the data by evaluating missing values, special symbols, word count, and other criteria.
    total_cells = df.size
    total_nulls = df.isnull().sum().sum()
    total_null_percentage = total_nulls / total_cells
    if total_null_percentage >= 0.2:
        return False

    specific_value = ['.', '#', 'null', 'NULL', '-', '*']
    specific_value_count = 0
    for val in specific_value:
        specific_value_count += (df == val).sum().sum()
    if specific_value_count >= 0.2:
        return False

    word_count_within_limit = df.applymap(check_word_count)
    if word_count_within_limit.any().any():
        return False

    if df.shape[1] <= 3:
        return False

    return True

def load_single_data(table_file, auto_feature_type, is_label=False, is_classify=False, seed=42, core_size=10000):
    # Load a single dataset and perform preprocessing based on the task type (classification or regression).
    if os.path.exists(table_file):
        logger.debug("Loading dataset from %s", table_file)
        df = pd.read_csv(table_file)

        if is_classify:
            target = df.columns.tolist()[-1]

            value_counts = df[target].value_counts()
            unique_values = value_counts[value_counts == 1].index
            df = df[~df[target].isin(unique_values)]

            y = df[target]
            X = df.drop([target], axis=1)

            # Downsample the data if too large
            if (X.shape[0] > core_size):
                sample_ratio = (core_size / X.shape[0])
                X, _, y, _ = train_test_split(X, y, train_size=sample_ratio, random_state=seed, stratify=y, shuffle=True)

        else:
            X = df
            if df.shape[0] > core_size:
                X = df.sample(n=core_size, random_state=seed)
            if is_label:
                target = df.columns.tolist()[-1]
                X = X.drop([target], axis=1)
            y = None

        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        if X.shape[1] > 1000:
            raise RuntimeError('Too many features!')

        # Divide features into categorical, binary, and numeric
        # Split into training and validation sets
        if is_classify:
            train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        else:
            train_dataset, test_dataset = train_test_split(X, test_size=0.2, random_state=seed, shuffle=True)
            y_train = None
            y_test = None
        if not check_data_quality(train_dataset):
            raise RuntimeError('Data quality is too poor!')
        preprocessor = TablePreprocessor(auto_feature_type=auto_feature_type).fit(train_dataset, y_train)
        cat_cols = preprocessor.categorical_columns
        num_cols = preprocessor.numerical_columns
        bin_cols = preprocessor.binary_columns
        train_dataset = preprocessor.transform(train_dataset)
        test_dataset = preprocessor.transform(test_dataset)
        y_train = preprocessor.transform_target(y_train)
        y_test = preprocessor.transform_target(y_test)
    
        assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
        logger.debug(
            "Loaded dataset rows=%s features=%s categorical=%s binary=%s numerical=%s",
            len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols),
        )
        return (train_dataset, y_train), (test_dataset, y_test), cat_cols, num_cols, bin_cols
    else:
        raise RuntimeError('No such data!')
    
def load_single_data_all(table_file, target=None, auto_feature_type=None, encode_cat=False, return_meta_frame=False, task_type='cls', preprocess=True, return_preprocessor=False):
    if os.path.exists(table_file):
        logger.debug("Loading dataset from %s", table_file)
        df = pd.read_csv(table_file)

        if not target:
            target = df.columns.tolist()[-1]
        if not auto_feature_type:
            auto_feature_type = Feature_type_recognition()


        df.dropna(axis=0, subset=[target], inplace=True)

        if task_type == 'cls':
            # Delete samples whose class has only one row, so stratified splits stay valid.
            count_num = list(df[target].value_counts())
            count_value = list(df[target].value_counts().index)
            delete_index = []
            for i,cnt in enumerate(count_num):
                if cnt <= 1:
                    index = df.loc[df[target]==count_value[i]].index.to_list()
                    delete_index.extend(index)
            df.drop(delete_index, axis=0, inplace=True)

        y = df[target]
        X = df.drop([target], axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = [], [], []
        preprocessor = None
        
        if task_type == 'reg':
            y = pd.to_numeric(y, errors='coerce').replace([np.inf, -np.inf], np.nan)
            valid_index = y.dropna().index
            X = X.loc[valid_index]
            y = pd.Series(y.loc[valid_index], index=valid_index, name=target)
        else:
            # encode target label
            y = pd.Series(y, index=X.index, name=target)

        if preprocess:
            preprocessor = TablePreprocessor(task_type=task_type, encode_cat=encode_cat, auto_feature_type=auto_feature_type).fit(X, y)
            cat_cols = preprocessor.categorical_columns
            bin_cols = preprocessor.binary_columns
            num_cols = preprocessor.numerical_columns
            y = preprocessor.transform_target(y)

        df_for_meta = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    else:
        raise RuntimeError('no such data file!')

    # start processing features
    # process num
    # process cate
    # X[cat_cols] = X[cat_cols].astype(str).str.lower()
    if preprocess:
        X = preprocessor.transform(X)

    assert not preprocess or len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    logger.debug(
        "Loaded dataset rows=%s features=%s categorical=%s binary=%s numerical=%s",
        len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols),
    )
    result = (X, y, cat_cols, num_cols, bin_cols)
    if return_meta_frame:
        result += (df_for_meta,)
    if return_preprocessor:
        result += (preprocessor,)
    return result

def load_single_data_for_pretrain(table_file, auto_feature_type, is_label=True, is_classify=True, seed=42, core_size=10000, return_preprocessor=False):
    """
    Load a single labeled dataset for pre-training. Returns training and test datasets along with feature column information.
    """
    if not os.path.exists(table_file):
        raise RuntimeError(f'No such data: {table_file}!')

    logger.debug("Loading dataset from %s", table_file)
    df = pd.read_csv(table_file)

    # Extract target column
    target = df.columns.tolist()[-1]
    y = df[target]
    y.name = target  # Ensure y has a name
    X = df.drop(columns=[target])
    if not is_classify:
        y = pd.to_numeric(y, errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    if is_classify:
        # Remove rows with target class having unique values
        value_counts = y.value_counts()
        unique_values = value_counts[value_counts == 1].index
        mask = ~y.isin(unique_values)
        X = X[mask]
        y = y[mask]

        # Downsample if necessary
        if X.shape[0] > core_size:
            sample_ratio = core_size / X.shape[0]
            X, _, y, _ = train_test_split(X, y, train_size=sample_ratio, random_state=seed, stratify=y, shuffle=True)
        
        # Encode the target variable
    else:
        # For regression tasks, downsample if necessary
        if X.shape[0] > core_size:
            X, _, y, _ = train_test_split(X, y, train_size=core_size, random_state=seed, shuffle=True)
            y.name = target

        # Normalize the target variable for regression

    # Convert column names to lowercase
    all_cols = [col.lower() for col in X.columns]
    X.columns = all_cols
    attribute_names = all_cols

    if X.shape[1] > 1000:
        raise RuntimeError('Too many features!')

    # Divide into categorical, binary, and numeric features
    # Split into training and validation sets
    if is_classify:
        train_dataset, test_dataset, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
    else:
        train_dataset, test_dataset, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True)
    if not check_data_quality(train_dataset):
        raise RuntimeError('Data quality is too poor!')
    preprocessor = TablePreprocessor(
        task_type='cls' if is_classify else 'reg',
        auto_feature_type=auto_feature_type,
        scale_target=not is_classify,
    ).fit(train_dataset, y_train)
    cat_cols = preprocessor.categorical_columns
    bin_cols = preprocessor.binary_columns
    num_cols = preprocessor.numerical_columns
    y_train = preprocessor.transform_target(y_train)
    y_test = preprocessor.transform_target(y_test)
    df_for_meta = pd.concat([train_dataset.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    train_dataset = preprocessor.transform(train_dataset)
    test_dataset = preprocessor.transform(test_dataset)
    
    # Ensure target has name
    if y_train is not None and y_train.name is None:
        y_train.name = target
    if y_test is not None and y_test.name is None:
        y_test.name = target

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    logger.debug(
        "Loaded dataset rows=%s features=%s categorical=%s binary=%s numerical=%s",
        len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols),
    )

    result = ((train_dataset, y_train), (test_dataset, y_test), cat_cols, num_cols, bin_cols, df_for_meta)
    if return_preprocessor:
        result += (preprocessor,)
    return result

def load_all_label_data_for_pretrain(label_data_path=None, seed=42, limit=10000, core_size=10000, is_classify=True, return_preprocessors=False):
    """
    Load all labeled datasets for pre-training, return formatted data for training and validation.
    """
    num_col_list, cat_col_list, bin_col_list = [], [], []
    train_data, val_data = [], []
    dataset_paths = []
    preprocessors = {}
    auto_feature_type = Feature_type_recognition()

    label_data_list = sorted(os.listdir(label_data_path))
    table_flag = 0

    # Load each dataset
    for data in tqdm(label_data_list, desc="Loading labeled data for pretrain"):
        if data.endswith(".csv"):
            data_path = os.path.join(label_data_path, data)

            try:
                if is_classify:
                    (X_train, y_train), (X_val, y_val), cat_cols, num_cols, bin_cols, df_all, preprocessor = load_single_data_for_pretrain(
                        table_file=data_path,
                        auto_feature_type=auto_feature_type,
                        is_label=True,
                        is_classify=True, 
                        seed=seed,
                        core_size=core_size,
                        return_preprocessor=True,
                    )
                else:
                    (X_train, y_train), (X_val, y_val), cat_cols, num_cols, bin_cols, df_all, preprocessor = load_single_data_for_pretrain(
                        table_file=data_path,
                        auto_feature_type=auto_feature_type,
                        is_label=True,
                        is_classify=False, 
                        seed=seed,
                        core_size=core_size,
                        return_preprocessor=True,
                    )
                y_train = y_train.dropna()
                X_train = X_train.loc[y_train.index]  # Synchronize filtering based on y_train index
                y_val = y_val.dropna()
                X_val = X_val.loc[y_val.index]

                # Use one stable full-table meta feature for train/validation splits.
                task_type = 'cls' if is_classify else 'reg'
                meta_features = build_table_meta_features(
                    df_all,
                    task_type=task_type,
                    categorical_columns=cat_cols,
                    numerical_columns=num_cols,
                )

                # Add column info and path
                num_col_list.append(num_cols)
                cat_col_list.append(cat_cols)
                bin_col_list.append(bin_cols)
                dataset_paths.append(data_path)

                # Store in the format (X, y, meta_features, table_flag)
                train_data.append(((X_train, y_train, meta_features), table_flag))
                val_data.append(((X_val, y_val, meta_features), table_flag))
                preprocessors[table_flag] = preprocessor
                table_flag += 1

            except Exception as e:
                logger.warning("Skipped %s due to error: %s", data, e)
                continue

            if len(train_data) >= limit:
                break

    logger.info("Loaded %s labeled datasets for pretraining.", len(train_data))
    result = (train_data, val_data, cat_col_list, num_col_list, bin_col_list, dataset_paths)
    if return_preprocessors:
        result += (preprocessors,)
    return result
