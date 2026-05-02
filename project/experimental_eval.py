import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.preprocess import PruningTransformer
from src.features import FeatureEngineeringTransformer, ClusteringTransformer

def add_all_enriched_features(train_df, test_df):
    print(">>> Starting Full Fusion Feature Enrichment...")
    
    # 1. Time-Series Momentum (Weather)
    # We combine them temporarily to calculate chronological rolling metrics
    full = pd.concat([train_df, test_df]).sort_values('TransactionDT')
    full['rolling_fraud_momentum'] = full['isFraud'].shift(1).rolling(window=1000, min_periods=1).mean()
    full['rolling_fraud_momentum'] = full['rolling_fraud_momentum'].fillna(full['isFraud'].mean())
    
    # 2. PCA (Latent Structure)
    # Scale and apply PCA on numeric columns
    scaler = StandardScaler()
    num_cols = full.select_dtypes(include=[np.number]).columns.drop(['isFraud', 'TransactionID'], errors='ignore')
    scaled_data = scaler.fit_transform(full[num_cols].fillna(0))
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(scaled_data)
    for i in range(3):
        full[f'pca_{i+1}'] = pca_feats[:, i]
        
    # 3. Association Rules (Heuristics)
    # We'll add a simple 'high_risk_product' feature based on known high-fraud categories from EDA
    # ProductCD 'W' and 'C' often have different profiles
    full['is_high_risk_product'] = full['ProductCD'].map({'W': 0, 'C': 1, 'H': 1, 'R': 0, 'S': 0}).fillna(0)

    # Split back
    train_enriched = full[full['TransactionID'].isin(train_df['TransactionID'])].sort_values('TransactionID')
    test_enriched = full[full['TransactionID'].isin(test_df['TransactionID'])].sort_values('TransactionID')
    
    X_train = train_enriched.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    X_test = test_enriched.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    y_train = train_enriched['isFraud']
    y_test = test_enriched['isFraud']
    
    return X_train, X_test, y_train, y_test

def run_experiment():
    print("Loading Data...")
    train = pd.read_csv('data/train_unbalanced.csv')
    test = pd.read_csv('data/test.csv')
    
    # Baseline
    print("\n--- Training Baseline Model ---")
    y_train_b = train['isFraud']
    X_train_b = train.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    y_test_b = test['isFraud']
    X_test_b = test.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    
    scale_weight = (len(y_train_b) - sum(y_train_b)) / sum(y_train_b)
    
    baseline_pipe = Pipeline([
        ('prune', PruningTransformer()),
        ('fe', FeatureEngineeringTransformer()),
        ('clustering', ClusteringTransformer(n_clusters=5)),
        ('model', XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, 
                               scale_pos_weight=scale_weight, random_state=42, n_jobs=-1))
    ])
    
    baseline_pipe.fit(X_train_b, y_train_b)
    b_probs = baseline_pipe.predict_proba(X_test_b)[:, 1]
    b_auc_pr = average_precision_score(y_test_b, b_probs)
    print(f"Baseline AUC-PR: {b_auc_pr:.4f}")
    
    # Full Fusion
    print("\n--- Training Full Fusion Model (PCA + TS + Rules) ---")
    X_train_f, X_test_f, y_train_f, y_test_f = add_all_enriched_features(train, test)
    
    fusion_pipe = Pipeline([
        ('prune', PruningTransformer()),
        ('fe', FeatureEngineeringTransformer()),
        ('clustering', ClusteringTransformer(n_clusters=5)),
        ('model', XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, 
                               scale_pos_weight=scale_weight, random_state=42, n_jobs=-1))
    ])
    
    fusion_pipe.fit(X_train_f, y_train_f)
    f_probs = fusion_pipe.predict_proba(X_test_f)[:, 1]
    f_auc_pr = average_precision_score(y_test_f, f_probs)
    print(f"Full Fusion AUC-PR: {f_auc_pr:.4f}")
    
    improvement = ((f_auc_pr - b_auc_pr) / b_auc_pr) * 100
    print(f"\nResult: Total Combined Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    run_experiment()


