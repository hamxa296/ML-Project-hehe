from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif


def load_data(train_path: str = "../data/merged_raw_train.csv", test_path: str | None = None):
    print("Loading Raw Dataset...")
    train = pd.read_csv(train_path)

    if "isFraud" not in train.columns:
        raise ValueError("Expected target column 'isFraud' in the merged raw training file.")

    train = train.sort_values([col for col in ["TransactionDT", "TransactionID"] if col in train.columns]).reset_index(drop=True)

    y = train["isFraud"].astype(int)
    X = train.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    if "TransactionDT" in X.columns:
        ordered_index = X["TransactionDT"].sort_values(kind="mergesort").index
        X = X.loc[ordered_index].reset_index(drop=True)
        y = y.loc[ordered_index].reset_index(drop=True)

    n_rows = len(X)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)

    X_train = X.iloc[:train_end].reset_index(drop=True)
    y_train = y.iloc[:train_end].reset_index(drop=True)
    X_val = X.iloc[train_end:val_end].reset_index(drop=True)
    y_val = y.iloc[train_end:val_end].reset_index(drop=True)
    X_test = X.iloc[val_end:].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)

    print(f"Split sizes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


class PruningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold: float = 0.95, corr_threshold: float = 0.985, top_k_features: int = 200, sample_size: int = 60000):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.top_k_features = top_k_features
        self.sample_size = sample_size
        self.keep_cols = None
        self.input_schema = {}
        self.feature_scores_ = {}
         
    def fit(self, X, y):
        print("Fitting PruningTransformer...")
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        y_series = pd.Series(y, index=X_df.index)
        self.input_schema = X_df.dtypes.to_dict()
        
        # 1. Missing Value Filter
        missing_pct = X_df.isnull().mean()
        X_tmp = X_df.loc[:, missing_pct <= self.missing_threshold]
         
        if X_tmp.empty:
            self.keep_cols = list(X_df.columns)
            return self

        # 2. Zero-Variance
        selector = VarianceThreshold(threshold=0)
        selector.fit(X_tmp.fillna(0))
        X_tmp = X_tmp.loc[:, selector.get_support()]
         
        # 3. Correlation filter on a manageable sample
        sample_frame = X_tmp.fillna(0)
        if len(sample_frame) > self.sample_size:
            sample_frame = sample_frame.sample(self.sample_size, random_state=42)
        corr_matrix = sample_frame.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        X_tmp = X_tmp.drop(columns=to_drop, errors='ignore')
         
        # 4. Information Gain Filter
        print("Calculating Information Gain...")
        sample_tr = X_tmp.fillna(0)
        if len(sample_tr) > self.sample_size:
            sample_tr = sample_tr.sample(self.sample_size, random_state=42)
        sample_y = y_series.loc[sample_tr.index]
        importances = mutual_info_classif(sample_tr, sample_y, random_state=42)

        feat_importances = pd.Series(importances, index=sample_tr.columns).sort_values(ascending=False)
        self.feature_scores_ = feat_importances.to_dict()
        top_k = feat_importances.nlargest(min(self.top_k_features, len(feat_importances))).index

        self.keep_cols = list(top_k)
        print(f"Baseline Feature Set Size after pruning: {len(self.keep_cols)}")
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        X_out = X_out.apply(pd.to_numeric, errors='coerce')

        if self.keep_cols is None:
            raise ValueError("PruningTransformer must be fit before transform.")

        for col in self.keep_cols:
            if col not in X_out.columns:
                X_out[col] = 0.0

        X_out = X_out[self.keep_cols].fillna(0.0)
        return X_out
