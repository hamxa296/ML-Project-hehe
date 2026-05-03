import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Re-using discretization logic from association.py logic for consistency
def _discretize_for_association(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    if 'ProductCD' in df.columns:
        d['ProductCD'] = 'ProductCD=' + df['ProductCD'].astype(str).str.strip()
    for col in ['card4', 'card6']:
        if col in df.columns:
            d[col] = col + '=' + df[col].astype(str).str.strip()
    if 'P_emaildomain' in df.columns:
        top_domains = df['P_emaildomain'].value_counts().nlargest(6).index.tolist()
        d['P_emaildomain'] = df['P_emaildomain'].apply(
            lambda x: f"email={x}" if x in top_domains else "email=other"
        )
    if 'TransactionAmt' in df.columns:
        amt = pd.to_numeric(df['TransactionAmt'], errors='coerce').fillna(0)
        bins = [0, 50, 200, float('inf')]
        labels = ['Amt=Low', 'Amt=Mid', 'Amt=High']
        d['AmtBin'] = pd.cut(amt, bins=bins, labels=labels, right=True).astype(str)
    return d.fillna('Unknown')

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
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        
    def fit(self, X, y=None):
        X_num = X.select_dtypes(include=[np.number])
        self.kmeans.fit(X_num.fillna(0))
        return self

    def transform(self, X):
        X_out = X.copy()
        X_num = X.select_dtypes(include=[np.number])
        X_out['cluster_label'] = self.kmeans.predict(X_num.fillna(0))
        return X_out

class PCATransformer(BaseEstimator, TransformerMixin):
    """
    Adds Principal Components as features.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.scaler = StandardScaler()
        self.num_cols = None

    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(self.num_cols) < self.n_components:
            self.n_components = len(self.num_cols)
            self.pca = PCA(n_components=self.n_components, random_state=42)
            
        X_scaled = self.scaler.fit_transform(X[self.num_cols].fillna(0))
        self.pca.fit(X_scaled)
        return self

    def transform(self, X):
        X_out = X.copy()
        X_scaled = self.scaler.transform(X[self.num_cols].fillna(0))
        X_pca = self.pca.transform(X_scaled)
        
        for i in range(self.n_components):
            X_out[f'pca_comp_{i+1}'] = X_pca[:, i]
            
        return X_out

class AssociationTransformer(BaseEstimator, TransformerMixin):
    """
    Uses Association Rule Mining patterns to create a 'risk_score' feature.
    """
    def __init__(self, min_support=0.01, min_confidence=0.05, top_n=10):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.top_n = top_n
        self.rules = []

    def fit(self, X, y=None):
        if y is None:
            return self
            
        try:
            from mlxtend.frequent_patterns import fpgrowth, association_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            print("mlxtend not installed, skipping AssociationTransformer fit")
            return self

        # Discretize and add target for mining
        df_disc = _discretize_for_association(X)
        df_disc['isFraud'] = y.map({0: 'isFraud=No', 1: 'isFraud=Yes'})
        
        # Sample for speed if large
        if len(df_disc) > 50000:
            df_disc = df_disc.sample(50000, random_state=42)
            
        records = df_disc.astype(str).values.tolist()
        te = TransactionEncoder()
        te_array = te.fit_transform(records)
        basket_df = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = fpgrowth(basket_df, min_support=self.min_support, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=self.min_confidence)
            # Filter for fraud-consequent rules
            fraud_rules = rules_df[rules_df['consequents'].apply(lambda x: 'isFraud=Yes' in x)]
            fraud_rules = fraud_rules.sort_values('lift', ascending=False).head(self.top_n)
            
            self.rules = [set(r) for r in fraud_rules['antecedents'].tolist()]
            
        return self

    def transform(self, X):
        X_out = X.copy()
        if not self.rules:
            X_out['association_risk_score'] = 0
            return X_out
            
        df_disc = _discretize_for_association(X)
        
        # Vectorized calculation of rule matches
        risk_scores = np.zeros(len(df_disc))
        
        # Pre-identify which column each item in a rule belongs to
        # Items are formatted as "ColName=Value" or custom prefixes like "email=..."
        col_map = {
            'ProductCD=': 'ProductCD',
            'card4=': 'card4',
            'card6=': 'card6',
            'email=': 'P_emaildomain',
            'Amt=': 'AmtBin'
        }
        
        for rule in self.rules:
            # A rule is a set of items, e.g., {'ProductCD=C', 'card6=credit'}
            rule_mask = np.ones(len(df_disc), dtype=bool)
            for item in rule:
                found_col = None
                for prefix, col in col_map.items():
                    if item.startswith(prefix):
                        found_col = col
                        break
                
                if found_col and found_col in df_disc.columns:
                    rule_mask &= (df_disc[found_col] == item)
            
            risk_scores += rule_mask.astype(int)
            
        X_out['association_risk_score'] = risk_scores
        return X_out
