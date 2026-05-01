#!/usr/bin/env python3
"""
Quick diagnostic to understand feature count discrepancy (339 vs 167 target)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path('.').resolve()))

from fraud_ensemble_paper.src.paper_pipeline import train_paper_style

print("=" * 80)
print("FEATURE ENGINEERING DIAGNOSTIC")
print("=" * 80)

# Load data
data_path = 'data/merged_raw_train.csv'
print(f"\n1. Loading data from: {data_path}")
raw = pd.read_csv(data_path, nrows=5000)
raw = raw.sort_values("TransactionDT").reset_index(drop=True)

# Baseline features (everything except ID columns)
X = raw.drop(columns=["isFraud", "TransactionID"], errors="ignore")
y = raw["isFraud"].astype(int)

print(f"   Raw baseline features: {X.shape[1]} columns")
print(f"   Expected: 431 baseline features")
print(f"   Match: {X.shape[1] == 431}")

# Check what features are in the data
v_cols = [c for c in X.columns if c.startswith('V')]
print(f"\n2. Feature breakdown:")
print(f"   V-features: {len(v_cols)}")
print(f"   Other features: {X.shape[1] - len(v_cols)}")

# Now run paper pipeline and see what happens
print(f"\n3. Running paper pipeline on 5000 samples...")

# Import and run the pipeline function directly
from fraud_ensemble_paper.src.paper_pipeline import train_paper_style

result = train_paper_style('data/merged_raw_train.csv', sample=5000, n_folds=2, verbose=False)

print(f"   Final features: {result['feature_count']}")
print(f"   Expected: 167 (per paper)")
print(f"   Gap: +{result['feature_count'] - 167}")

# Let's also manually check the preprocessing pipeline
print(f"\n4. Manual preprocessing pipeline check:")

from fraud_ensemble_paper.src.paper_pipeline import _categorical_columns, _numeric_columns, _v_columns

# Load fresh
raw2 = pd.read_csv(data_path, nrows=5000)
raw2 = raw2.sort_values("TransactionDT").reset_index(drop=True)
X2 = raw2.drop(columns=["isFraud", "TransactionID"], errors="ignore")
y2 = raw2["isFraud"].astype(int)

train_end = int(0.7 * len(X2))
X_train, y_train = X2.iloc[:train_end], y2.iloc[:train_end]

print(f"   Pre-engineering: {X_train.shape[1]} features")

# Apply feature engineering
from fraud_ensemble_paper.src.paper_pipeline import FeatureEngineeringTransformer

fe = FeatureEngineeringTransformer()
X_train_fe = fe.fit_transform(X_train, y_train)

print(f"   Post-engineering: {X_train_fe.shape[1]} features")

# Apply pruning
from fraud_ensemble_paper.src.paper_pipeline import PruningTransformer

pruner = PruningTransformer(
    missing_threshold=0.95,
    corr_threshold=0.98,
    info_gain_threshold=0.001,
)
X_train_pruned = pruner.fit_transform(X_train_fe, y_train)

print(f"   Post-pruning: {X_train_pruned.shape[1]} features")
print(f"   Expected: 167 features (per paper)")

# Check pruning details
if hasattr(pruner, 'selected_features_'):
    print(f"\n5. Pruning details:")
    print(f"   Selected features: {len(pruner.selected_features_)}")
    
if hasattr(pruner, 'pruning_log_'):
    print(f"\n6. Pruning log:")
    for stage, count in pruner.pruning_log_.items():
        print(f"   {stage}: {count} features")
else:
    print(f"\n6. No pruning log available - check PruningTransformer implementation")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
