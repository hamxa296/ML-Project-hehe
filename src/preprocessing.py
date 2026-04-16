import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def reduce_mem_usage(df, verbose=True):
    """
    ### What this step does:
    Iterates through all columns of a dataframe and modifies the data type
    to reduce memory usage significantly.
    
    ### Why this is necessary:
    The Kaggle IEEE Fraud Detection dataset is massive (~1.5GB combined). 
    Loading it into pandas using default data types (like float64 or int64) 
    can easily crash a standard computer's memory or slow down training drastically.
    By finding the minimum required data type (e.g., int8, float16) for each column,
    we can often reduce memory size by 50-70%.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

class DataMerger(BaseEstimator, TransformerMixin):
    """
    ### What this step does:
    A Scikit-Learn custom Transformer that merges the Transaction and Identity 
    datasets using a Left Join on the 'TransactionID'. It also standardizes the 
    identity column names.
    
    ### Why this is necessary:
    In the raw dataset, transactions and related identity data are separated.
    Additionally, the identity columns in the test set use dashes (e.g., 'id-01') 
    while the training set uses underscores ('id_01'). This discrepancy will crash 
    the model during inference if not handled properly.
    """
    def __init__(self, identity_df):
        self.identity_df = identity_df
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Merging Transaction and Identity data...")
        # Standardize column naming convention
        self.identity_df.columns = [col.replace('-', '_') for col in self.identity_df.columns]
        
        # Left merge to keep all transactions, appending identity data where it exists
        merged_df = X.merge(self.identity_df, on='TransactionID', how='left')
        return merged_df

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    ### What this step does:
    Extracts chronological features from the 'TransactionDT' column.
    
    ### Why this is necessary:
    'TransactionDT' is provided as a timedelta from an unknown reference date. 
    ML algorithms cannot derive patterns from raw monotonic counters very easily.
    By converting it into cyclical features like 'hour of day' or 'day of week',
    we expose temporal patterns typical of fraud (e.g., fraud happens more often at 3 AM).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Extracting time-based features from TransactionDT...")
        X_copy = X.copy()
        
        # We assume TransactionDT is in seconds
        # 86400 seconds in a day. We can estimate the time of day.
        # Adding an arbitrary reference point is fine for cyclical extraction
        X_copy['Transaction_Day'] = np.floor((X_copy['TransactionDT'] / (3600 * 24) - 1) % 7)
        X_copy['Transaction_Hour'] = np.floor(X_copy['TransactionDT'] / 3600) % 24
        
        return X_copy


class DropHighNulls(BaseEstimator, TransformerMixin):
    """
    ### What this step does:
    Drops columns that contain missing values above a certain threshold (default 80%).
    
    ### Why this is necessary:
    Many columns (like the V-features) are extremely sparse. If a column is missing 
    95% of its data, any imputation logic we try to apply will be heavily biased or 
    meaningless. Removing these reduces dimensionality and prevents noise, avoiding the 'curse of dimensionality'.
    """
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        null_fractions = X.isnull().mean()
        self.cols_to_drop_ = null_fractions[null_fractions > self.threshold].index.tolist()
        return self

    def transform(self, X):
        print(f"Dropping {len(self.cols_to_drop_)} columns with over {self.threshold * 100}% nulls...")
        return X.drop(columns=self.cols_to_drop_, errors='ignore')


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    ### What this step does:
    Encodes categorical features by replacing the categories with their frequency 
    (i.e., how many times they appear in the dataset).
    
    ### Why this is necessary:
    Standard label encoding just assigns arbitrary integers which can mislead 
    tree algorithms. One-hot encoding would explode the memory due to high cardinality. 
    Frequency encoding provides a meaningful numeric signal: outliers and rare categories 
    get distinct, low values, which is often highly predictive of fraud.
    """
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.frequency_maps_ = {}

    def fit(self, X, y=None):
        # If no columns provided, find all objects/categories
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.cat_cols:
            if col in X.columns:
                # Store the frequencies normalized (as percentages)
                freq = X[col].value_counts(dropna=False, normalize=True).to_dict()
                self.frequency_maps_[col] = freq
        return self

    def transform(self, X):
        print(f"Applying frequency encoding to {len(self.frequency_maps_)} categorical columns...")
        X_copy = X.copy()
        for col, freq_map in self.frequency_maps_.items():
            if col in X_copy.columns:
                # Map categories to their fitted frequencies, fill new/unseen with 0
                X_copy[col] = X_copy[col].map(freq_map).fillna(0)
        return X_copy


# =========================================================================
# Orchestration: Full Pipeline Execution
# =========================================================================
# When run directly, this script will:
#   1. Load the raw Transaction and Identity CSVs
#   2. Apply memory optimization to both
#   3. Pass through the full preprocessing pipeline
#   4. Save the result to data/processed_train.csv for EDA
if __name__ == "__main__":
    import os
    import sys
    from sklearn.pipeline import Pipeline

    # -----------------------------------------------------------------------
    # Configuration: Paths (works whether run from project root or src/)
    # -----------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_PATH = os.path.join(DATA_DIR, "processed_train.csv")

    print("=" * 60)
    print("  IEEE-CIS Fraud Detection — Preprocessing Pipeline")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Load raw data
    # -----------------------------------------------------------------------
    print("\n[1/5] Loading raw CSV files (this may take a minute)...")
    train_transaction = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
    train_identity    = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
    print(f"      train_transaction shape : {train_transaction.shape}")
    print(f"      train_identity shape    : {train_identity.shape}")

    # -----------------------------------------------------------------------
    # Step 2: Memory Optimisation (applied before pipeline to save RAM)
    # -----------------------------------------------------------------------
    print("\n[2/5] Optimising memory usage...")
    train_transaction = reduce_mem_usage(train_transaction)
    train_identity    = reduce_mem_usage(train_identity)

    # -----------------------------------------------------------------------
    # Step 3: Build & execute the scikit-learn Pipeline
    # -----------------------------------------------------------------------
    print("\n[3/5] Building and running the preprocessing pipeline...")
    pipeline = Pipeline([
        ("merger",       DataMerger(identity_df=train_identity)),
        ("time_extract", TimeFeatureExtractor()),
        ("drop_nulls",   DropHighNulls(threshold=0.85)),
        ("freq_encoder", FrequencyEncoder()),
    ])

    processed = pipeline.fit_transform(train_transaction)

    # -----------------------------------------------------------------------
    # Step 4: Final numeric cast — ensure every column is numeric
    # -----------------------------------------------------------------------
    print("\n[4/5] Final type-cast: converting any remaining object columns...")
    for col in processed.select_dtypes(include=["object"]).columns:
        processed[col] = pd.to_numeric(processed[col], errors="coerce")

    print(f"      Final processed shape: {processed.shape}")
    print(f"      isFraud distribution :\n{processed['isFraud'].value_counts()}")

    # -----------------------------------------------------------------------
    # Step 5: Save to CSV
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Saving processed dataset to:\n      {OUTPUT_PATH}")
    processed.to_csv(OUTPUT_PATH, index=False)
    print("\n✅  Done! processed_train.csv is ready for EDA.")
    print("=" * 60)
