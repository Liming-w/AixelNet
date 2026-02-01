import os 
import pdb
import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        print(f'load from local data dir {table_file}')
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

            y = LabelEncoder().fit_transform(y.values)
            y = pd.Series(y, index=X.index)
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

        if not check_data_quality(X):
            raise RuntimeError('Data quality is too poor!')

        # Divide features into categorical, binary, and numeric
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        if len(cat_cols) > 0:
            for col in cat_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)       
            X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower())
        if len(num_cols) > 0:
            for col in num_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)       
            X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])
        
        # Split into training and validation sets
        if is_classify:
            train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        else:
            train_dataset, test_dataset = train_test_split(X, test_size=0.2, random_state=seed, shuffle=True)
            y_train = None
            y_test = None
    
        assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols)))
        return (train_dataset, y_train), (test_dataset, y_test), cat_cols, num_cols, bin_cols
    else:
        raise RuntimeError('No such data!')
    
def load_single_data_all(table_file, target=None, auto_feature_type=None, encode_cat=False):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)

        if not target:
            target = df.columns.tolist()[-1]
        if not auto_feature_type:
            auto_feature_type = Feature_type_recognition()


        # Delete the sample whose label count is 1 or label is nan
        count_num = list(df[target].value_counts())
        count_value = list(df[target].value_counts().index)
        delete_index = []
        for i,cnt in enumerate(count_num):
            if cnt <= 1:
                index = df.loc[df[target]==count_value[i]].index.to_list()
                delete_index.extend(index)
        df.drop(delete_index, axis=0, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, subset=[target], inplace=True)

        y = df[target]
        X = df.drop([target], axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        
        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y, index=X.index, name=target)
    else:
        raise RuntimeError('no such data file!')

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            # X[cat_cols] = X[cat_cols].astype(str).str.lower()
            X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower()) 

    if len(bin_cols) > 0:
        for col in bin_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        for col in bin_cols:
            if X[col].nunique() <= 1:
                raise RuntimeError('bin feature process error!')
    
    X = X[bin_cols + num_cols + cat_cols]

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
    return X, y, cat_cols, num_cols, bin_cols

def load_single_data_for_pretrain(table_file, auto_feature_type, is_label=True, is_classify=True, seed=42, core_size=10000):
    """
    Load a single labeled dataset for pre-training. Returns training and test datasets along with feature column information.
    """
    if not os.path.exists(table_file):
        raise RuntimeError(f'No such data: {table_file}!')

    print(f'load from local data dir {table_file}')
    df = pd.read_csv(table_file)

    # Extract target column
    target = df.columns.tolist()[-1]
    y = df[target]
    y.name = target  # Ensure y has a name
    X = df.drop(columns=[target])

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
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y, index=X.index, name=target)
    else:
        # For regression tasks, downsample if necessary
        if X.shape[0] > core_size:
            X, _, y, _ = train_test_split(X, y, train_size=core_size, random_state=seed, shuffle=True)
            y.name = target

        # Normalize the target variable for regression
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        y = pd.Series(y, index=X.index, name=target)

    # Convert column names to lowercase
    all_cols = [col.lower() for col in X.columns]
    X.columns = all_cols
    attribute_names = all_cols

    if X.shape[1] > 1000:
        raise RuntimeError('Too many features!')

    if not check_data_quality(X):
        raise RuntimeError('Data quality is too poor!')

    # Divide into categorical, binary, and numeric features
    cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
    if len(cat_cols) > 0:
        for col in cat_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower())
    if len(num_cols) > 0:
        for col in num_cols:
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    # Split into training and validation sets
    if is_classify:
        train_dataset, test_dataset, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
    else:
        train_dataset, test_dataset, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True)
    
    # Ensure target has name
    if y_train is not None and y_train.name is None:
        y_train.name = target
    if y_test is not None and y_test.name is None:
        y_test.name = target

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols)))

    return (train_dataset, y_train), (test_dataset, y_test), cat_cols, num_cols, bin_cols

def load_all_label_data_for_pretrain(label_data_path=None, seed=42, limit=10000, core_size=10000, is_classify=True):
    """
    Load all labeled datasets for pre-training, return formatted data for training and validation.
    """
    num_col_list, cat_col_list, bin_col_list = [], [], []
    train_data, val_data = [], []
    dataset_paths = []
    auto_feature_type = Feature_type_recognition()

    label_data_list = os.listdir(label_data_path)
    table_flag = 0

    # Load each dataset
    for data in tqdm(label_data_list, desc="Loading labeled data for pretrain"):
        if data.endswith(".csv"):
            data_path = os.path.join(label_data_path, data)

            try:
                if is_classify:
                    (X_train, y_train), (X_val, y_val), cat_cols, num_cols, bin_cols = load_single_data_for_pretrain(
                        table_file=data_path,
                        auto_feature_type=auto_feature_type,
                        is_label=True,
                        is_classify=True, 
                        seed=seed,
                        core_size=core_size,
                    )
                else:
                    (X_train, y_train), (X_val, y_val), cat_cols, num_cols, bin_cols = load_single_data_for_pretrain(
                        table_file=data_path,
                        auto_feature_type=auto_feature_type,
                        is_label=True,
                        is_classify=False, 
                        seed=seed,
                        core_size=core_size,
                    )
                y_train = y_train.dropna()
                X_train = X_train.loc[y_train.index]  # Synchronize filtering based on y_train index
                y_val = y_val.dropna()
                X_val = X_val.loc[y_val.index]

                # Combine into DataFrame
                df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
                df_val = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

                # Add column info and path
                num_col_list.append(num_cols)
                cat_col_list.append(cat_cols)
                bin_col_list.append(bin_cols)
                dataset_paths.append(data_path)

                # Store in the format (X, y, df, table_flag)
                train_data.append(((X_train, y_train, df_train), table_flag))
                val_data.append(((X_val, y_val, df_val), table_flag))
                table_flag += 1

            except Exception as e:
                print(f"Skipped {data} due to error: {e}")
                continue

            if len(train_data) >= limit:
                break

    print(f"Loaded {len(train_data)} labeled datasets for pretraining.")
    return train_data, val_data, cat_col_list, num_col_list, bin_col_list, dataset_paths