import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

def load_data(train_path='../data/train_unbalanced.csv', test_path='../data/test.csv'):
    print("Loading Raw Datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    y_train = train['isFraud']
    y_test = test['isFraud']
    X_train = train.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    X_test = test.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    
    print(f"Initial Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test

class PruningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keep_cols = None
        self.input_schema = {}
        
    def fit(self, X, y):
        print("Fitting PruningTransformer...")
        # Save training schema for robust inference validation
        self.input_schema = X.dtypes.to_dict()
        
        # 1. Missing Value Filter (>95%)
        missing_pct = X.isnull().mean()
        X_tmp = X.loc[:, missing_pct <= 0.95]
        
        # 2. Zero-Variance
        selector = VarianceThreshold(threshold=0)
        selector.fit(X_tmp.fillna(0))
        X_tmp = X_tmp.loc[:, selector.get_support()]
        
        # 3. Correlation (>0.98)
        corr_matrix = X_tmp.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
        X_tmp = X_tmp.drop(columns=to_drop, errors='ignore')
        
        # 4. Information Gain Filter (Keep top 167)
        print("Calculating Information Gain...")
        sample_tr = X_tmp.fillna(-1).sample(min(50000, len(X_tmp)), random_state=42)
        sample_y = y.loc[sample_tr.index]
        importances = mutual_info_classif(sample_tr, sample_y)
        
        feat_importances = pd.Series(importances, index=X_tmp.columns)
        top_167_cols = feat_importances.nlargest(167).index
        
        self.keep_cols = list(top_167_cols)
        print(f"Baseline Feature Set Size after pruning: {len(self.keep_cols)}")
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Robust real-world input handling
        # 1. Missing columns: Add with default values
        for col, dtype in self.input_schema.items():
            if col not in X_out.columns:
                X_out[col] = 0 if np.issubdtype(dtype, np.number) else ""
                
        # 2. Type inconsistencies: Cast safely
        for col, dtype in self.input_schema.items():
            try:
                X_out[col] = X_out[col].astype(dtype)
            except Exception:
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0)
                
        # 3. Extra/unexpected columns: Drop safely (by strictly selecting keep_cols)
        # 4. Incorrect column order: Realign to match training schema
        return X_out[self.keep_cols]
