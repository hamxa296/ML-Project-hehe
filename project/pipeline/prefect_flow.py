import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

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
from src.train_baseline import train_baseline_model, evaluate_baseline
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
    """EDA on data AFTER the full preprocessing pipeline has been applied."""
    print("\n>>> Running Processed Data EDA...")
    from src.eda_processed import run_processed_eda
    steps = pipeline.named_steps
    X_t = steps['prune'].transform(X_train)
    X_t = steps['fe'].transform(X_t)
    X_t = steps['clustering'].transform(X_t)
    feature_names = list(X_t.columns)
    X_array = X_t.values
    out_dir = ARTIFACTS_DIR / 'eda_processed'
    run_processed_eda(
        X_transformed=X_array,
        y=y_train.reset_index(drop=True),
        feature_names=feature_names,
        out_dir=out_dir,
        raw_shape=X_train.shape,
    )
    print(f">>> Processed EDA complete. Plots -> {out_dir}")


# ── Core ML Pipeline Tasks ────────────────────────────────────────────────────

@task(retries=2, name="Load Data")
def load_task():
    if os.getenv("CI"):
        # In CI, use the smaller dev dataset that is checked into Git
        train_path = PROJECT_ROOT / 'data' / 'dev.csv'
        test_path  = PROJECT_ROOT / 'data' / 'dev.csv'
    else:
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
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "version":         version,
        "model_type":      "XGBClassifier (w/ KMeans)",
        "hyperparameters": json.dumps(interesting_hyperparams),
        "precision":       prec,
        "recall":          rec,
        "auc_roc":         auc_roc,
        "auc_pr":          auc_pr,
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


# ── Phase 1: Analytical ML Tasks ─────────────────────────────────────────────

@task(name="Regression — Transaction Velocity Forecasting")
def regression_task(train_df: pd.DataFrame):
    """Ridge regression: predict next-window fraud count from lag + time features."""
    from src.regression import run_regression
    graphs_dir = PROJECT_ROOT / 'results' / 'graphs'
    return run_regression(train_df, artifacts_dir=ARTIFACTS_DIR, graphs_dir=graphs_dir)


@task(name="Time Series — Fraud Rate Analysis")
def timeseries_task(train_df: pd.DataFrame):
    """Descriptive time series: rolling fraud rates, z-score anomaly detection, heatmap."""
    from src.timeseries import run_timeseries
    graphs_dir = PROJECT_ROOT / 'results' / 'graphs'
    return run_timeseries(train_df, artifacts_dir=ARTIFACTS_DIR, graphs_dir=graphs_dir)


@task(name="Dimensionality Reduction — PCA")
def dimensionality_reduction_task(pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """PCA on processed 175-feature space — 2D scatter plot, variance explained."""
    from src.dimensionality_reduction import run_dimensionality_reduction
    steps = pipeline.named_steps
    X_t = steps['prune'].transform(X_train)
    X_t = steps['fe'].transform(X_t)
    X_t = steps['clustering'].transform(X_t)
    feature_names = list(X_t.columns)
    graphs_dir = PROJECT_ROOT / 'results' / 'graphs'
    return run_dimensionality_reduction(
        X_t.values, y_train, feature_names,
        artifacts_dir=ARTIFACTS_DIR, graphs_dir=graphs_dir
    )


@task(name="Clustering — Behaviour Profile Analysis")
def clustering_analysis_task(pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """Profile KMeans clusters: fraud rate per cluster, feature centroids."""
    from src.clustering_analysis import run_clustering_analysis
    graphs_dir = PROJECT_ROOT / 'results' / 'graphs'
    return run_clustering_analysis(pipeline, X_train, y_train,
                                   artifacts_dir=ARTIFACTS_DIR, graphs_dir=graphs_dir)


@task(name="Association Rules — Fraud Pattern Mining")
def association_task(train_df: pd.DataFrame):
    """FPGrowth on categorical features — finds fraud-consequent association rules."""
    from src.association import run_association
    graphs_dir = PROJECT_ROOT / 'results' / 'graphs'
    return run_association(train_df, artifacts_dir=ARTIFACTS_DIR, graphs_dir=graphs_dir)


@task(name="Baseline Experiment")
def train_baseline_task(X_train, X_test, y_train, y_test, version):
    """Trains a simple baseline to compare against the main champion model."""
    pipeline = train_baseline_model(X_train, y_train)
    metrics = evaluate_baseline(pipeline, X_test, y_test)
    
    results_path = ARTIFACTS_DIR / 'results.csv'
    res_df = pd.DataFrame([{
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "version":         version,
        "model_type":      "LogisticRegression (Baseline)",
        "hyperparameters": "{}",
        "precision":       0.0,
        "recall":          0.0,
        "auc_roc":         metrics['auc_roc'],
        "auc_pr":          metrics['auc_pr'],
    }])
    if results_path.exists():
        res_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(results_path, index=False)
        
    print(f">>> Baseline evaluation complete. AUC-PR: {metrics['auc_pr']:.4f}")
    return metrics


@task(name="Evidently ML Health Report")
def evidently_report_task(X_train, X_test, y_train, y_test, pipeline):
    """Generates a deep ML health report using Evidently."""
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    
    # We use a sample of data for the report to keep it lightweight
    ref_data = X_train.copy().sample(min(1000, len(X_train)))
    ref_data['target'] = y_train.loc[ref_data.index]
    ref_data['prediction'] = pipeline.predict_proba(ref_data.drop(columns='target'))[:, 1]
    
    curr_data = X_test.copy().sample(min(1000, len(X_test)))
    curr_data['target'] = y_test.loc[curr_data.index]
    curr_data['prediction'] = pipeline.predict_proba(curr_data.drop(columns='target'))[:, 1]
    
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    
    report.run(reference_data=ref_data, current_data=curr_data)
    report_path = ARTIFACTS_DIR / 'evidently_report.html'
    report.save_html(str(report_path))
    print(f">>> Evidently report generated at {report_path}")


# ── Main Flow ─────────────────────────────────────────────────────────────────

from pipeline.notify import send_discord

def notify_success(flow, flow_run, state):
    send_discord(f"Pipeline {flow_run.name} succeeded! View dashboard for metrics.", color=3066993, title="✅ Pipeline Success")

def notify_failure(flow, flow_run, state):
    send_discord(f"Pipeline {flow_run.name} failed! Error: {state.message}", color=15158332, title="❌ Pipeline Failed")

@flow(name="Robust Fraud Detection Unified Pipeline", on_completion=[notify_success], on_failure=[notify_failure])
def training_pipeline(config: dict = {"min_auc_pr": 0.6}):
    version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Step 1: Load raw data
    X_train, X_test, y_train, y_test = load_task()

    # If running in CI (GitHub Actions), we sample the data to avoid OOM/Timeouts
    if os.getenv("CI"):
        print(">>> CI Environment detected: Sampling 5% of data for smoke test...")
        X_train = X_train.sample(frac=0.05, random_state=42)
        y_train = y_train.loc[X_train.index]
        X_test = X_test.sample(frac=0.05, random_state=42)
        y_test = y_test.loc[X_test.index]

    # Full raw train DataFrame needed by analytical tasks (they need isFraud column)
    if os.getenv("CI"):
        train_path_str = str(PROJECT_ROOT / 'data' / 'dev.csv')
    else:
        train_path_str = str(PROJECT_ROOT / 'data' / 'train_unbalanced.csv')
    train_df_raw   = pd.read_csv(train_path_str)

    # Step 2: Raw EDA
    raw_eda_task(train_path_str)

    # Step 2.5: Baseline Experiment (Roadmap Phase 4)
    train_baseline_task(X_train, X_test, y_train, y_test, version)

    # Step 3: Train the full sklearn pipeline (Champion Model)
    pipeline = train_pipeline_task(X_train, y_train)

    # Step 4: Processed EDA
    processed_eda_task(pipeline, X_train, y_train)

    # Step 4.5: Evidently ML Health Report (Roadmap Phase 2)
    evidently_report_task(X_train, X_test, y_train, y_test, pipeline)

    # Step 5: Evaluate + quality gate
    auc_pr = evaluate_and_log_task(pipeline, X_test, y_test, version)

    if auc_pr < config.get("min_auc_pr", 0.6):
        raise ValueError(f"Quality gate failed: AUC-PR {auc_pr:.4f} < {config['min_auc_pr']}")

    # Step 6: Save model
    save_model_task(pipeline, version)

    # ── Phase 1: Analytical branches (run after training) ────────────────────
    regression_task(train_df_raw)
    timeseries_task(train_df_raw)
    dimensionality_reduction_task(pipeline, X_train, y_train)
    clustering_analysis_task(pipeline, X_train, y_train)
    association_task(train_df_raw)


if __name__ == "__main__":
    training_pipeline()
