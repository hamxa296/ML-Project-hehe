import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Feature Engineering for the production pipeline.
    Handles temporal features, log transforms, and basic scaling.
    """
    def __init__(self):
        self.medians = None
        
    def fit(self, X, y=None):
        self.medians = X.median()
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. Temporal Features (matching the analysis tasks)
        if 'TransactionDT' in X_out.columns:
            X_out['hour'] = (X_out['TransactionDT'] // 3600) % 24
            X_out['day_of_week'] = (X_out['TransactionDT'] // (3600 * 24)) % 7
            
        # 2. Log Transform for Amount
        if 'TransactionAmt' in X_out.columns:
            X_out['Amt_Log'] = np.log1p(X_out['TransactionAmt'])
            
        # 3. Simple Imputation
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
