import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Anchor to the project root (project/) regardless of CWD
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
ARTIFACTS_DIR.mkdir(exist_ok=True)
sys.path.append(str(PROJECT_ROOT))

from prefect import flow, task
import joblib
import pandas as pd
from src.preprocess import load_data, PruningTransformer
from src.features import FeatureEngineeringTransformer, ClusteringTransformer
from src.train import train_model
from src.predict import predict
from src.evaluate import evaluate_model
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline


# ── EDA Tasks ────────────────────────────────────────────────────────────────

@task(name="Raw Data EDA")
def raw_eda_task(train_path: str):
    """Full EDA on the raw unprocessed training CSV."""
    print("\n>>> Running Raw Data EDA...")
    from src.eda_raw import run_raw_eda
    out_dir = ARTIFACTS_DIR / 'eda_raw'
    run_raw_eda(train_path=train_path, out_dir=out_dir)
    print(f">>> Raw EDA complete. Plots -> {out_dir}")


@task(name="Processed Data EDA")
def processed_eda_task(pipeline, X_train, y_train):
    """
    EDA on data AFTER the full preprocessing pipeline has been applied.
    Calls each fitted transformer step directly by name to avoid sklearn's
    Pipeline.check_is_fitted() issue when pipelines are deserialized by Prefect.
    """
    print("\n>>> Running Processed Data EDA...")
    from src.eda_processed import run_processed_eda

    # Access each fitted step directly by name — no Pipeline wrapper involved
    steps = pipeline.named_steps   # OrderedDict: {'prune': ..., 'fe': ..., 'clustering': ..., 'model': ...}
    X_t = steps['prune'].transform(X_train)
    X_t = steps['fe'].transform(X_t)
    X_t = steps['clustering'].transform(X_t)
    # X_t is now a DataFrame with cluster_label appended

    # Feature names come directly from the DataFrame columns
    feature_names = list(X_t.columns)
    X_array = X_t.values   # convert to numpy for the EDA module

    out_dir = ARTIFACTS_DIR / 'eda_processed'
    run_processed_eda(
        X_transformed=X_array,
        y=y_train.reset_index(drop=True),
        feature_names=feature_names,
        out_dir=out_dir,
        raw_shape=X_train.shape,
    )
    print(f">>> Processed EDA complete. Plots -> {out_dir}")



# ── ML Pipeline Tasks ─────────────────────────────────────────────────────────

@task(retries=2, name="Load Data")
def load_task():
    train_path = PROJECT_ROOT / 'data' / 'train_unbalanced.csv'
    test_path  = PROJECT_ROOT / 'data' / 'test.csv'
    return load_data(str(train_path), str(test_path))


@task(name="Train Model")
def train_pipeline_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Evaluate and Log")
def evaluate_and_log_task(pipeline, X_test, y_test, version):
    probs, preds = predict(pipeline, X_test)

    auc_pr  = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)
    prec    = precision_score(y_test, preds)
    rec     = recall_score(y_test, preds)

    model_params = pipeline.named_steps['model'].get_params()
    interesting_hyperparams = {
        "n_estimators":  model_params.get("n_estimators"),
        "max_depth":     model_params.get("max_depth"),
        "learning_rate": model_params.get("learning_rate"),
    }

    results_path = ARTIFACTS_DIR / 'results.csv'
    res_df = pd.DataFrame([{
        "timestamp":      datetime.now().isoformat(),
        "version":        version,
        "model_type":     "XGBClassifier (w/ KMeans)",
        "hyperparameters": json.dumps(interesting_hyperparams),
        "precision":      prec,
        "recall":         rec,
        "auc_roc":        auc_roc,
        "auc_pr":         auc_pr,
    }])

    if results_path.exists():
        res_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(results_path, index=False)

    evaluate_model(y_test, probs, preds,
                   project_root=PROJECT_ROOT,
                   version=version,
                   artifacts_dir=ARTIFACTS_DIR)

    print("\n--- PIPELINE METRICS FOR CI/CD ---")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print("----------------------------------\n")
    return auc_pr


@task(name="Save Model")
def save_model_task(pipeline, version):
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, models_dir / f"model_{version}.pkl")
    joblib.dump(pipeline, models_dir / "model_latest.pkl")
    print(f"Pipeline saved as models/model_{version}.pkl and model_latest.pkl")


# ── Main Flow ─────────────────────────────────────────────────────────────────

@flow(name="Robust Fraud Detection Unified Pipeline")
def training_pipeline(config: dict = {"min_auc_pr": 0.6}):
    version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Step 1: Load raw data
    X_train, X_test, y_train, y_test = load_task()

    # Step 2: Raw EDA — runs on the full unprocessed training data
    train_path = str(PROJECT_ROOT / 'data' / 'train_unbalanced.csv')
    raw_eda_task(train_path)

    # Step 3: Train the full sklearn pipeline
    pipeline = train_pipeline_task(X_train, y_train)

    # Step 4: Processed EDA — runs on the transformed training data
    processed_eda_task(pipeline, X_train, y_train)

    # Step 5: Evaluate on test set + quality gate
    auc_pr = evaluate_and_log_task(pipeline, X_test, y_test, version)

    if auc_pr < config.get("min_auc_pr", 0.6):
        raise ValueError(f"Quality gate failed: AUC-PR {auc_pr:.4f} < {config['min_auc_pr']}")

    # Step 6: Save model only if quality gate passed
    save_model_task(pipeline, version)


if __name__ == "__main__":
    training_pipeline()
