import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from datetime import datetime, timezone
import json
import joblib
from pathlib import Path

def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, version: str, artifacts_dir: Path):
    print(">>> Training Baseline Logistic Regression...")
    
    # Simple imputation and scaling for baseline
    # No complex feature engineering or clustering
    X_train_clean = X_train.fillna(0)
    X_test_clean = X_test.fillna(0)
    
    # Optional scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled)
    
    auc_pr = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    
    results_path = artifacts_dir / 'results.csv'
    res_df = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "model_type": "LogisticRegression_baseline",
        "hyperparameters": json.dumps({"max_iter": 1000}),
        "precision": prec,
        "recall": rec,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }])
    
    if results_path.exists():
        res_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(results_path, index=False)
        
    print(f"--- BASELINE METRICS ---")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print("------------------------\n")
    
    return {
        "precision": prec,
        "recall": rec,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr
    }
