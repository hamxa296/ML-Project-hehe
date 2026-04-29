from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import FeatureEngineeringTransformer
from src.modeling import FraudEnsembleModel, build_meta_baseline_model
from src.preprocess import PruningTransformer


def _build_preprocessor():
    return Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),
        (
            "pruning",
            PruningTransformer(
                missing_threshold=0.95,
                corr_threshold=0.985,
                top_k_features=200,
                sample_size=60000,
            ),
        ),
    ])


def _build_base_models(scale_pos_weight: float):
    xgb_primary = XGBClassifier(
        n_estimators=1400,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    xgb_regularized = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    logistic_baseline = build_meta_baseline_model()

    return {
        "xgb_primary": xgb_primary,
        "xgb_regularized": xgb_regularized,
        "logistic_baseline": logistic_baseline,
    }


def train_model(X_train, y_train, X_val, y_val):
    y_train = np.asarray(y_train).astype(int)
    y_val = np.asarray(y_val).astype(int)
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = neg / pos

    preprocessor = _build_preprocessor()
    base_models = _build_base_models(scale_pos_weight)
    ensemble = FraudEnsembleModel(preprocessor=preprocessor, base_models=base_models)

    print("Training the fraud ensemble pipeline...")
    ensemble.fit(X_train, y_train, X_val, y_val)
    print("Validation summary:", ensemble.summary())
    return ensemble


def save_model(model, output_dir="models", version="latest"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    joblib.dump(model, output_path / f"model_{version}.pkl")
    joblib.dump(model, output_path / "model_latest.pkl")
    return output_path / f"model_{version}.pkl"
