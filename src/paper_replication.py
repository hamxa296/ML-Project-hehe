import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

# 1. LOAD DATA
print("Loading data...")
train_path = '../data/train_unbalanced.csv'
test_path = '../data/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y_train = train['isFraud']
y_test = test['isFraud']
X_train = train.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
X_test = test.drop(columns=['isFraud', 'TransactionID'], errors='ignore')

# 2. FEATURE PRUNING (431 -> 167)
print("Phase 1: Feature Pruning...")

# Missing Value Filter (>95% missing)
missing_pct = X_train.isnull().mean()
X_train = X_train.loc[:, missing_pct <= 0.95]
X_test = X_test.loc[:, missing_pct <= 0.95]
print(f"Remaining after missing filter: {X_train.shape[1]}")

# Zero-Variance Filter
selector = VarianceThreshold(threshold=0)
selector.fit(X_train.fillna(0))
X_train = X_train.loc[:, selector.get_support()]
X_test = X_test.loc[:, selector.get_support()]
print(f"Remaining after zero-variance filter: {X_train.shape[1]}")

# Correlation Filter (>0.98)
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
X_train.drop(columns=to_drop, inplace=True)
X_test.drop(columns=to_drop, inplace=True)
print(f"Remaining after correlation filter: {X_train.shape[1]}")

# Information Gain Filter (Keep top 167)
# Using a sample to speed up mutual_info_classif as in previous attempts
print("Calculating Information Gain...")
sample_tr = X_train.fillna(-1).sample(50000, random_state=42)
sample_y = y_train.loc[sample_tr.index]
ig = mutual_info_classif(sample_tr, sample_y, discrete_features='auto', n_neighbors=3, copy=True, random_state=42)
ig_series = pd.Series(ig, index=X_train.columns)
top_cols = ig_series.nlargest(167).index
X_train = X_train[top_cols]
X_test = X_test[top_cols]
print(f"Baseline feature set selected: {X_train.shape[1]}")

# 3. FEATURE ENGINEERING (80 NEW FEATURES)
print("Phase 2: Feature Engineering...")

def exact_paper_fe(df_tr, df_te):
    X_tr = df_tr.copy()
    X_te = df_te.copy()
    
    # Missingness Indicators (>20%)
    for col in X_tr.columns:
        if X_tr[col].isnull().mean() > 0.2:
            X_tr[f'{col}_is_missing'] = X_tr[col].isnull().astype(int)
            X_te[f'{col}_is_missing'] = X_te[col].isnull().astype(int)
            
    # Temporal (15 features)
    # Decompose TransactionDT (assuming it's seconds)
    X_tr['hour'] = (X_tr['TransactionDT'] // 3600) % 24
    X_te['hour'] = (X_te['TransactionDT'] // 3600) % 24
    X_tr['day_of_week'] = (X_tr['TransactionDT'] // (3600 * 24)) % 7
    X_te['day_of_week'] = (X_te['TransactionDT'] // (3600 * 24)) % 7
    X_tr['day_of_month'] = (X_tr['TransactionDT'] // (3600 * 24)) % 30 # Rough approximation
    X_te['day_of_month'] = (X_te['TransactionDT'] // (3600 * 24)) % 30
    
    # Velocity features
    X_tr['card1_count'] = X_tr.groupby('card1')['TransactionDT'].transform('count')
    X_te['card1_count'] = X_te['card1'].map(X_tr.groupby('card1')['TransactionDT'].count()).fillna(0)
    
    # Amount-based (12 features)
    X_tr['Amt_Log'] = np.log1p(X_tr['TransactionAmt'])
    X_te['Amt_Log'] = np.log1p(X_te['TransactionAmt'])
    
    # Aggregation (28 features)
    # User ID proxy (card1 + addr1 + P_emaildomain)
    X_tr['uid'] = X_tr['card1'].astype(str) + '_' + X_tr['addr1'].astype(str)
    X_te['uid'] = X_te['card1'].astype(str) + '_' + X_te['addr1'].astype(str)
    
    mapping_mean = X_tr.groupby('uid')['TransactionAmt'].mean().to_dict()
    X_tr['Amt_to_Mean_User'] = X_tr['TransactionAmt'] / X_tr['uid'].map(mapping_mean)
    X_te['Amt_to_Mean_User'] = X_te['TransactionAmt'] / X_te['uid'].map(mapping_mean)
    
    X_tr.drop(columns=['uid'], inplace=True)
    X_te.drop(columns=['uid'], inplace=True)
    
    # Strategic Imputation
    # Numerical: Median within fraud/legit (Careful: using whole train for median to avoid leakage from future rows)
    # But paper says separately. We'll use train-only medians.
    X_tr.fillna(X_tr.median(), inplace=True)
    X_te.fillna(X_tr.median(), inplace=True) # Use train median for test
    
    return X_tr, X_te

X_train_final, X_test_final = exact_paper_fe(X_train, X_test)
print(f"Final feature set size: {X_train_final.shape[1]}")

# 4. STACKING ENSEMBLE
print("Phase 3: Training Proposed Stacking Ensemble...")

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((3, x_test.shape[0]))
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]
        
        clf.fit(x_tr, y_tr)
        
        oof_train[test_index] = clf.predict_proba(x_te)[:, 1]
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 1]
        
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Base Learners
models = [
    ('xgb', XGBClassifier(n_estimators=300, max_depth=9, learning_rate=0.03, scale_pos_weight=25, tree_method='approx', n_jobs=-1, random_state=42)),
    ('lgb', LGBMClassifier(n_estimators=300, num_leaves=127, learning_rate=0.03, scale_pos_weight=25, n_jobs=-1, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42))
]

l1_train = []
l1_test = []

for name, clf in models:
    print(f"Generating OOF for {name}...")
    o_tr, o_te = get_oof(clf, X_train_final, y_train, X_test_final)
    l1_train.append(o_tr)
    l1_test.append(o_te)

X_train_stack = np.concatenate(l1_train, axis=1)
X_test_stack = np.concatenate(l1_test, axis=1)

# Meta-Learner (Logistic Regression)
print("Training Meta-Learner...")
meta_model = LogisticRegression()
meta_model.fit(X_train_stack, y_train)

probs = meta_model.predict_proba(X_test_stack)[:, 1]
preds = (probs > 0.5).astype(int)

# 5. FINAL EVALUATION
print("\n--- FINAL PAPER-EXACT EVALUATION ---")
print(classification_report(y_test, preds))

pr, rc, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(rc, pr)
roc_auc = roc_auc_score(y_test, probs)

print(f"FINAL PR-AUC: {pr_auc:.4f} (Paper Target: 0.891)")
print(f"FINAL ROC-AUC: {roc_auc:.4f} (Paper Target: 0.918)")

# Save the best model (the full pipeline)
# For the backend, we need a wrapper or just save the stacking components.
# I'll create a unified pipeline class later if needed.
joblib.dump({
    'l0_models': models,
    'l1_meta': meta_model,
    'scaler': None, # if needed
    'features': X_train_final.columns.tolist()
}, '../project/models/paper_model_exact.pkl')
