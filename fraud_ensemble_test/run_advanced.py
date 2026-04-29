"""Orchestrator to run advanced experiments covering velocity, entity graphs, tuning, stacking, calibration, and monitoring.

This runner is intentionally conservative: each heavy step is run only if dependencies are present.
"""
import sys
from pathlib import Path
import warnings
import numpy as np

# Add fraud_ensemble_test to path
sys.path.insert(0, str(Path(__file__).parent))

# Conditional imports with graceful fallbacks
try:
    from src.features import FeatureEngineeringTransformer
    from src.preprocess import PruningTransformer
    from src.modeling import FraudEnsembleModel, build_meta_baseline_model
except ImportError as e:
    print(f"Error importing core modules: {e}")
    sys.exit(1)

try:
    from src.optuna_tuner import tune_xgb
except Exception:
    tune_xgb = None

try:
    from src.imbalance import apply_smote
except Exception:
    apply_smote = None

try:
    from src.stacking import train_stack
except Exception:
    train_stack = None

try:
    from src.calibration import platt_calibration, cv_isotonic_calibration
except Exception:
    platt_calibration = None

try:
    from src.monitoring import should_retrain
except Exception:
    should_retrain = None

import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, precision_score, recall_score, classification_report
import joblib


def main(data_path: str = "data/merged_raw_train.csv"):
    root = Path(__file__).parent.parent
    data_path = root / data_path
    
    print("=" * 90)
    print("🚀 ADVANCED FRAUD ENSEMBLE - FULL PIPELINE TEST")
    print("=" * 90)
    print(f"\n📂 Loading data from: {data_path}")
    
    raw = pd.read_csv(data_path)
    raw = raw.sort_values("TransactionDT").reset_index(drop=True)
    X = raw.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    y = raw["isFraud"].astype(int)
    
    print(f"   ✅ Loaded {len(raw):,} transactions, {len(X.columns)} features")
    print(f"   📊 Fraud rate: {y.mean()*100:.2f}%")

    # simple split
    n = len(raw)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    print(f"   Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    print("\n🔧 Fitting preprocessor (feature engineering + pruning)...")
    preprocessor = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("pruning", PruningTransformer(top_k_features=200)),
    ])
    preprocessor.fit(X_train, y_train)
    X_train_feat = preprocessor.transform(X_train)
    X_val_feat = preprocessor.transform(X_val)
    X_test_feat = preprocessor.transform(X_test)
    print(f"   ✅ Features after engineering+pruning: {X_train_feat.shape[1]}")

    # Optuna tuning (optional)
    print("\n🎯 Running optuna hyperparameter tuning (if available)...")
    best_params = None
    if tune_xgb is not None:
        try:
            sample_idx = np.random.choice(len(X_train_feat), min(20000, len(X_train_feat)), replace=False)
            best_params = tune_xgb(X_train_feat.iloc[sample_idx], y_train.iloc[sample_idx], n_trials=10)
            print(f"   ✅ Optuna best params: {best_params}")
        except Exception as e:
            print(f"   ⚠️  Optuna skipped: {e}")
    else:
        print("   ⚠️  optuna not available")

    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = neg / pos
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    base_models = {
        "xgb_primary": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='aucpr', n_jobs=1, random_state=42),
        "xgb_regularized": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='aucpr', n_jobs=1, random_state=42),
        "logistic_baseline": build_meta_baseline_model(),
    }

    # Apply SMOTE (optional)
    print("\n⚖️  Applying imbalance strategy (SMOTE if available)...")
    X_train_bal, y_train_bal = X_train_feat, y_train
    if apply_smote is not None:
        try:
            X_train_bal, y_train_bal = apply_smote(X_train_feat, y_train)
            print(f"   ✅ SMOTE applied: {len(X_train_bal):,} samples (from {len(X_train_feat):,})")
        except Exception as e:
            print(f"   ⚠️  SMOTE skipped: {e}")
    else:
        print("   ⚠️  imblearn.SMOTE not available")

    # Stacking (optional)
    print("\n🏗️  Training stacking base models with OOF predictions...")
    meta_X, meta_y, fitted_bases = None, None, None
    if train_stack is not None:
        try:
            meta_X, meta_y, fitted_bases = train_stack(base_models, X_train_bal, y_train_bal, n_folds=5)
            print(f"   ✅ Stacking OOF meta matrix shape: {meta_X.shape}")
            # Fit meta learner
            meta_learner = LogisticRegression(max_iter=2000, solver='lbfgs')
            meta_learner.fit(meta_X, meta_y)
            print(f"   ✅ Meta-learner trained")
        except Exception as e:
            print(f"   ⚠️  Stacking skipped: {e}")
            fitted_bases = None
    else:
        print("   ⚠️  Stacking not available")

    # Build ensemble
    print("\n🤖 Training base ensemble models...")
    ensemble = FraudEnsembleModel(preprocessor=preprocessor, base_models=base_models)
    ensemble.fit(X_train, y_train, X_val, y_val)
    print(f"   ✅ Ensemble trained")
    summary = ensemble.summary()
    print(f"   Base weights: {summary['base_weights']}")
    print(f"   Threshold: {summary['threshold']:.4f}")

    # Calibration (optional)
    print("\n📊 Calibrating ensemble predictions...")
    if platt_calibration is not None:
        try:
            val_probs = ensemble.predict_proba(X_val_feat)[:, 1]
            calib = platt_calibration(val_probs, y_val.values)
            ensemble.calibrator_ = calib
            print(f"   ✅ Platt calibration fitted")
        except Exception as e:
            print(f"   ⚠️  Calibration skipped: {e}")
    else:
        print("   ⚠️  Calibration modules not available")

    # Evaluate on test
    print("\n📈 Evaluating on test set ({:,} samples)...".format(len(X_test)))
    X_test_feat = preprocessor.transform(X_test)
    probs = ensemble.predict_proba(X_test_feat)[:, 1]
    preds = ensemble.predict(X_test_feat)
    
    pr_auc = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    
    print("\n" + "=" * 90)
    print("TEST SET PERFORMANCE METRICS")
    print("=" * 90)
    print(f"PR-AUC (Primary):        {pr_auc:.6f}  {'✅' if pr_auc > 0.5 else '❌' if pr_auc < 0.45 else '⚠️'}")
    print(f"ROC-AUC (Secondary):     {roc_auc:.6f}  {'✅' if roc_auc > 0.88 else '⚠️'}")
    print(f"Precision (Low FP):      {precision:.6f}  {'✅' if precision > 0.60 else '❌' if precision < 0.55 else '⚠️'}")
    print(f"Recall (Catch Fraud):    {recall:.6f}  {'✅' if recall > 0.50 else '❌' if recall < 0.35 else '⚠️'}")
    print("=" * 90)
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, preds, target_names=['Legitimate', 'Fraud']))

    # Monitoring (optional)
    print("\n📡 Computing drift metric between validation and test predictions...")
    if should_retrain is not None:
        try:
            val_probs_check = ensemble.predict_proba(X_val_feat)[:, 1]
            retrain_flag, drift_score = should_retrain(val_probs_check, probs)
            print(f"   Drift score: {drift_score:.4f}")
            print(f"   Should retrain: {retrain_flag}")
        except Exception as e:
            print(f"   ⚠️  Monitoring skipped: {e}")
    else:
        print("   ⚠️  Monitoring not available")

    # Save artifacts
    out_dir = root / "fraud_ensemble_test" / "models"
    out_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(ensemble, out_dir / "advanced_ensemble.pkl")
    print(f"\n💾 Saved advanced ensemble to: {out_dir / 'advanced_ensemble.pkl'}")

    print("\n" + "=" * 90)
    print("✨ ADVANCED PIPELINE COMPLETE!")
    print("=" * 90)


if __name__ == '__main__':
    main()

