import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.null_cols_to_flag = []
        self.mapping_median = {}
        self.user_counts = {}
        self.global_medians = {}

    def fit(self, X, y=None):
        X_tr = X.copy()
        
        # 1. Missingness indicators
        for col in X_tr.columns:
            if X_tr[col].isnull().mean() > 0.2:
                self.null_cols_to_flag.append(col)
                
        # Calculate aggregations temporarily
        if 'card1' in X_tr.columns and 'addr1' in X_tr.columns:
            X_tr['uid'] = X_tr['card1'].astype(str) + '_' + X_tr['addr1'].astype(str)
            if 'TransactionAmt' in X_tr.columns:
                self.mapping_median = X_tr.groupby('uid')['TransactionAmt'].median().to_dict()
            if 'TransactionDT' in X_tr.columns:
                self.user_counts = X_tr.groupby('uid')['TransactionDT'].count().to_dict()
            X_tr.drop(columns=['uid'], inplace=True)
            
        # Get final medians for the completely transformed schema
        X_tmp = self._apply_fe(X_tr.copy())
        self.global_medians = X_tmp.median().to_dict()
        return self

    def _apply_fe(self, X_df):
        X_out = X_df.copy()
        
        for col in self.null_cols_to_flag:
            if col in X_out.columns:
                X_out[f'{col}_null'] = X_out[col].isnull().astype(int)
        
        if 'TransactionDT' in X_out.columns:
            X_out['hour'] = (pd.to_numeric(X_out['TransactionDT'], errors='coerce').fillna(0) // 3600) % 24
            X_out['dow']  = (pd.to_numeric(X_out['TransactionDT'], errors='coerce').fillna(0) // (3600 * 24)) % 7
            
        if 'card1' in X_out.columns and 'addr1' in X_out.columns:
            uid = X_out['card1'].astype(str) + '_' + X_out['addr1'].astype(str)
            
            if 'TransactionAmt' in X_out.columns:
                amt = pd.to_numeric(X_out['TransactionAmt'], errors='coerce').fillna(0)
                med_series = uid.map(self.mapping_median).fillna(amt.median() if not amt.empty else 0)
                X_out['Amt_to_Median_User'] = amt / (med_series + 1e-6)  # avoid division by zero
                
            if 'TransactionDT' in X_out.columns:
                X_out['user_count'] = uid.map(self.user_counts).fillna(1)
            
        return X_out

    def transform(self, X):
        X_out = self._apply_fe(X)
        
        # Ensure exact column match with training schema and median impute missing
        for col, med in self.global_medians.items():
            if col not in X_out.columns:
                X_out[col] = med
            X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(med)
            
        # Drop unexpected columns dynamically and enforce strict ordering
        X_out = X_out[list(self.global_medians.keys())]
        return X_out

class ClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.features_to_cluster = None

    def fit(self, X, y=None):
        # Fit clustering only on training data (which is guaranteed to be clean and imputed)
        self.features_to_cluster = list(X.columns)
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        # Inference correctly utilizes the trained cluster model
        X_out = X.copy()
        clusters = self.kmeans.predict(X[self.features_to_cluster])
        X_out['cluster_label'] = clusters
        return X_out
