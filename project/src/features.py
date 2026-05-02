import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Feature Engineering for the production pipeline.
    Handles temporal features, missingness indicators, and user velocity.
    """
    def __init__(self):
        self.medians = None
        self.missingness_cols = []
        self.user_counts = {}
        self.user_medians = {}
        
    def fit(self, X, y=None):
        # 1. Missingness Indicators
        self.missingness_cols = [col for col in X.columns if X[col].isnull().mean() > 0.2]
        
        # 4. Strict Imputation medians
        self.medians = X.median(numeric_only=True)
        
        # 3. User-Level Velocity Aggregations
        if 'card1' in X.columns and 'addr1' in X.columns:
            uid = X['card1'].astype(str) + "_" + X['addr1'].astype(str)
            self.user_counts = uid.value_counts().to_dict()
            if 'TransactionAmt' in X.columns:
                df_tmp = pd.DataFrame({'uid': uid, 'Amt': X['TransactionAmt']})
                self.user_medians = df_tmp.groupby('uid')['Amt'].median().to_dict()
                
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. Missingness Indicators
        for col in self.missingness_cols:
            if col in X_out.columns:
                X_out[f'{col}_null'] = X_out[col].isnull().astype(int)
                
        # 2. Temporal Features
        if 'TransactionDT' in X_out.columns:
            X_out['hour'] = (X_out['TransactionDT'] // 3600) % 24
            X_out['dow'] = (X_out['TransactionDT'] // (3600 * 24)) % 7
            
        # 3. User-Level Velocity Aggregations
        if 'card1' in X_out.columns and 'addr1' in X_out.columns:
            uid = X_out['card1'].astype(str) + "_" + X_out['addr1'].astype(str)
            X_out['user_count'] = uid.map(self.user_counts).fillna(1)
            
            if 'TransactionAmt' in X_out.columns:
                user_med = uid.map(self.user_medians).fillna(X_out['TransactionAmt'].median())
                user_med = user_med.replace(0, 0.01) # Avoid division by zero
                X_out['Amt_to_Median_User'] = X_out['TransactionAmt'] / user_med

        # 4. Strict Imputation
        X_out.fillna(self.medians, inplace=True)
        X_out.fillna(0, inplace=True)
        
        return X_out

class ClusteringTransformer(BaseEstimator, TransformerMixin):
    """
    Adds a 'cluster_label' feature to the dataset using KMeans.
    Used for both the model and the standalone clustering analysis.
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        
    def fit(self, X, y=None):
        # We only cluster on numerical columns
        X_num = X.select_dtypes(include=[np.number])
        self.kmeans.fit(X_num.fillna(0))
        return self

    def transform(self, X):
        X_out = X.copy()
        X_num = X.select_dtypes(include=[np.number])
        X_out['cluster_label'] = self.kmeans.predict(X_num.fillna(0))
        return X_out
