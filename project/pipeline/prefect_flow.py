import sys
import os
import json
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prefect import flow, task
import joblib
import pandas as pd
from src.preprocess import load_data
from src.train import train_model
from src.predict import predict
from src.evaluate import evaluate_model
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score

@task(retries=2)
def load_task():
    base_dir = Path(__file__).resolve().parent.parent.parent
    train_path = os.path.join(base_dir, 'data', 'train_unbalanced.csv')
    test_path = os.path.join(base_dir, 'data', 'test.csv')
    return load_data(train_path, test_path)

@task
def train_pipeline_task(X_train, y_train):
    return train_model(X_train, y_train)

@task
def evaluate_and_log_task(pipeline, X_test, y_test, version):
    probs, preds = predict(pipeline, X_test)
    
    auc_pr = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    
    # Advanced Experiment Tracking Logging
    model_params = pipeline.named_steps['model'].get_params()
    interesting_hyperparams = {
        "n_estimators": model_params.get("n_estimators"),
        "max_depth": model_params.get("max_depth"),
        "learning_rate": model_params.get("learning_rate")
    }
    
    results_path = "results.csv"
    res_df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "model_type": "XGBClassifier (w/ KMeans)",
        "hyperparameters": json.dumps(interesting_hyperparams),
        "precision": prec,
        "recall": rec,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr
    }])
    
    if os.path.exists(results_path):
        res_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(results_path, index=False)
        
    evaluate_model(y_test, probs, preds)
    
    # Print explicit metrics for CI/CD Actions visibility
    print("\n--- PIPELINE METRICS FOR CI/CD ---")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print("----------------------------------\n")
    
    return auc_pr

@task
def save_model_task(pipeline, version):
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, f"models/model_{version}.pkl")
    joblib.dump(pipeline, "models/model_latest.pkl")
    print(f"Pipeline saved as models/model_{version}.pkl and model_latest.pkl")

@flow(name="Robust Fraud Detection Unified Pipeline")
def training_pipeline(config: dict = {"min_auc_pr": 0.6}):
    version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    X_train, X_test, y_train, y_test = load_task()
    pipeline = train_pipeline_task(X_train, y_train)
    
    # Model Quality Evaluation
    auc_pr = evaluate_and_log_task(pipeline, X_test, y_test, version)
    
    if auc_pr < config.get("min_auc_pr", 0.6):
        raise ValueError(f"Quality gate failed: AUC-PR {auc_pr:.4f} < {config['min_auc_pr']}")
        
    save_model_task(pipeline, version)

if __name__ == "__main__":
    training_pipeline()
