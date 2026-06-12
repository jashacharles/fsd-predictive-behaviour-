# this file organise csv inputs

import re
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_csv_files(data_dir):
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    return csv_files


def sort_visible_car_columns(cols):
    def key_fn(col):
        match = re.search(r"visible_car_(\d+)_(.+)", col)
        if match:
            return int(match.group(1)), match.group(2)
        return 999, col

    return sorted(cols, key=key_fn)


def get_feature_columns(sample_csv_path):
    sample_df = pd.read_csv(sample_csv_path)

    ego_columns = [
        "ego_lane",
        "ego_x",
        "ego_y",
        "ego_speed"
    ]

    visible_car_columns = [
        c for c in sample_df.columns
        if c.startswith("visible_car_")
        and not c.endswith("_behavior_id")
    ]

    visible_car_columns = sort_visible_car_columns(visible_car_columns)

    feature_columns = ego_columns + visible_car_columns

    return feature_columns


def build_sequences_from_csv(file_path, feature_columns, seq_len, predict_horizon=0):
    df = pd.read_csv(file_path)

    missing_columns = [c for c in feature_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"{file_path} is missing columns: {missing_columns}")

    if "movement_class" not in df.columns:
        raise ValueError(f"{file_path} does not contain movement_class column")

    MOVEMENT_CLASS_MAP = {"none": 0, "left": 1, "right": 2}

    X_raw = df[feature_columns].values.astype(np.float32)
    y_raw = df["movement_class"].map(MOVEMENT_CLASS_MAP).values.astype(np.int64)

    X_seq = []
    y_seq = []

    for i in range(len(df) - seq_len - predict_horizon):
        X_seq.append(X_raw[i:i + seq_len])
        y_seq.append(y_raw[i + seq_len + predict_horizon - 1])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def build_dataset_from_multiple_csvs(csv_files, feature_columns, seq_len, predict_horizon=0):
    all_X = []
    all_y = []

    for file_path in csv_files:
        X_seq, y_seq = build_sequences_from_csv(
            file_path=file_path,
            feature_columns=feature_columns,
            seq_len=seq_len,
            predict_horizon=predict_horizon,
        )

        all_X.append(X_seq)
        all_y.append(y_seq)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y


def normalize_train_data(X):
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6

    X_norm = (X - mean) / std

    return X_norm, mean, std


def apply_normalization(X, mean, std):
    return (X - mean) / std


def split_train_test(X, y, test_size=0.2):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


class DrivingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]