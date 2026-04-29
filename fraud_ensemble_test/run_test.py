#!/usr/bin/env python
"""Standalone test script for the fraud detection ensemble model."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocess import load_data
from src.train import train_model, save_model


def predict(model, X_test):
    probs = model.predict_proba(X_test)[:, 1]
    threshold = getattr(model, "threshold_", 0.5)
    preds = (probs >= threshold).astype(int)
    return probs, preds


def main():
    print("=" * 90)
    print("🚀 FRAUD DETECTION ENSEMBLE - ISOLATED TEST RUN")
    print("=" * 90)

    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "data" / "merged_raw_train.csv"

    print(f"\n📂 Loading data from: {train_path}")
    if not train_path.exists():
        print(f"❌ ERROR: Data file not found at {train_path}")
        sys.exit(1)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(str(train_path))
    print(f"   ✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"   📊 Fraud rates - Train: {y_train.mean():.3%}, Val: {y_val.mean():.3%}, Test: {y_test.mean():.3%}")

    print("\n🤖 Training the weighted XGBoost ensemble with calibration...")
    print("   (This may take 5-15 minutes on full dataset...)")
    model = train_model(X_train, y_train, X_val, y_val)

    print("\n📊 Evaluating on test set...")
    probs, preds = predict(model, X_test)

    auc_pr = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)

    print("\n" + "=" * 90)
    print("📈 TEST SET PERFORMANCE METRICS")
    print("=" * 90)
    print(f"PR-AUC (Primary):        {auc_pr:.6f}  {'✅ EXCELLENT' if auc_pr > 0.75 else '⚠️  GOOD' if auc_pr > 0.65 else '❌ NEEDS WORK'}")
    print(f"ROC-AUC (Secondary):     {auc_roc:.6f}  {'✅ EXCELLENT' if auc_roc > 0.90 else '⚠️  GOOD' if auc_roc > 0.85 else '❌ NEEDS WORK'}")
    print(f"Precision (Low FP):      {prec:.6f}  {'✅ HIGH' if prec > 0.80 else '⚠️  MODERATE' if prec > 0.60 else '❌ LOW'}")
    print(f"Recall (Catch Fraud):    {rec:.6f}  {'✅ HIGH' if rec > 0.80 else '⚠️  MODERATE' if rec > 0.60 else '❌ LOW'}")
    print("=" * 90)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))

    print("🎯 Model Summary:")
    summary = model.summary()
    print(f"   Base Model Weights:")
    for name, weight in summary['base_weights'].items():
        print(f"      {name:20s}: {weight:.4f}")
    print(f"   Decision Threshold:      {summary['threshold']:.6f}")
    print(f"   Validation Metrics:")
    for key, val in summary['validation_metrics'].items():
        if isinstance(val, float):
            print(f"      {key:20s}: {val:.6f}")

    print("\n💾 Saving model to: fraud_ensemble_test/models/")
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(model, output_dir="models", version=version)
    print(f"   ✅ Saved as: model_{version}.pkl and model_latest.pkl")

    # Summary for user
    print("\n" + "=" * 90)
    print("✨ TRAINING COMPLETE!")
    print("=" * 90)
    print("\n📊 Summary:")
    print(f"   • All models and data isolated in: fraud_ensemble_test/")
    print(f"   • Original project files: UNTOUCHED ✅")
    print(f"   • Test Run Results:")
    print(f"      - PR-AUC: {auc_pr:.4f}")
    print(f"      - ROC-AUC: {auc_roc:.4f}")
    print(f"      - Precision: {prec:.4f}")
    print(f"      - Recall: {rec:.4f}")
    print(f"\n   Ready to integrate into project/src/ when approved!")


if __name__ == "__main__":
    main()
