import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.preprocess import PruningTransformer
from src.features import FeatureEngineeringTransformer, ClusteringTransformer

def add_pca_features(X_train, X_test, n_components=3):
    print(f"Calculating PCA Features (n={n_components})...")
    # PCA requires scaling and handling missing values (Pruning handles missing)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]).fillna(0))
    X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]).fillna(0))
    
    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(X_train_scaled)
    pca_test = pca.transform(X_test_scaled)
    
    # Add to DFs
    for i in range(n_components):
        X_train[f'pca_{i+1}'] = pca_train[:, i]
        X_test[f'pca_{i+1}'] = pca_test[:, i]
    
    return X_train, X_test

def run_experiment():
    print("Loading Data...")
    train = pd.read_csv('data/train_unbalanced.csv')
    test = pd.read_csv('data/test.csv')
    
    y_train = train['isFraud']
    X_train_raw = train.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    y_test = test['isFraud']
    X_test_raw = test.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    
    # 1. Baseline Model
    print("\n--- Training Baseline Model ---")
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    baseline_pipe = Pipeline([
        ('prune', PruningTransformer()),
        ('fe', FeatureEngineeringTransformer()),
        ('clustering', ClusteringTransformer(n_clusters=5)),
        ('model', XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, 
                               scale_pos_weight=scale_weight, random_state=42, n_jobs=-1))
    ])
    
    baseline_pipe.fit(X_train_raw, y_train)
    b_probs = baseline_pipe.predict_proba(X_test_raw)[:, 1]
    b_auc_pr = average_precision_score(y_test, b_probs)
    print(f"Baseline AUC-PR: {b_auc_pr:.4f}")
    
    # 2. Enriched Model (PCA)
    print("\n--- Training Enriched Model (PCA) ---")
    # We apply PCA after basic cleaning but before the pipeline logic
    X_train_e, X_test_e = add_pca_features(X_train_raw.copy(), X_test_raw.copy())
    
    enriched_pipe = Pipeline([
        ('prune', PruningTransformer()),
        ('fe', FeatureEngineeringTransformer()),
        ('clustering', ClusteringTransformer(n_clusters=5)),
        ('model', XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, 
                               scale_pos_weight=scale_weight, random_state=42, n_jobs=-1))
    ])
    
    enriched_pipe.fit(X_train_e, y_train)
    e_probs = enriched_pipe.predict_proba(X_test_e)[:, 1]
    e_auc_pr = average_precision_score(y_test, e_probs)
    print(f"Enriched (PCA) AUC-PR: {e_auc_pr:.4f}")
    
    improvement = ((e_auc_pr - b_auc_pr) / b_auc_pr) * 100
    print(f"\nResult: Improvement of {improvement:.2f}% in AUC-PR")

if __name__ == "__main__":
    run_experiment()


