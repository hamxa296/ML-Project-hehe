from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def choose_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-9)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), {
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
    }


class FraudEnsembleModel:
    def __init__(self, preprocessor, base_models: dict, calibrator=None):
        self.preprocessor = preprocessor
        self.base_models = base_models
        self.calibrator = calibrator or IsotonicRegression(out_of_bounds="clip")
        self.fitted_base_models_ = {}
        self.base_weights_ = {}
        self.threshold_ = 0.5
        self.validation_metrics_ = {}

    def fit(self, X_train, y_train, X_val, y_val):
        self.preprocessor.fit(X_train, y_train)
        X_train_t = self.preprocessor.transform(X_train)
        X_val_t = self.preprocessor.transform(X_val)

        val_prob_matrix = {}
        self.fitted_base_models_ = {}

        for name, model in self.base_models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train_t, y_train)
            self.fitted_base_models_[name] = fitted_model
            val_prob_matrix[name] = fitted_model.predict_proba(X_val_t)[:, 1]

        weight_scores = {}
        for name, probs in val_prob_matrix.items():
            score = average_precision_score(y_val, probs)
            weight_scores[name] = max(score, 1e-6)

        total_weight = float(sum(weight_scores.values()))
        self.base_weights_ = {name: score / total_weight for name, score in weight_scores.items()}

        ensemble_val_raw = np.zeros(len(X_val_t), dtype=float)
        for name, probs in val_prob_matrix.items():
            ensemble_val_raw += self.base_weights_[name] * probs

        self.calibrator.fit(ensemble_val_raw, y_val)
        ensemble_val_calibrated = self.calibrator.transform(ensemble_val_raw)
        self.threshold_, threshold_metrics = choose_threshold(y_val, ensemble_val_calibrated)

        val_pred = (ensemble_val_calibrated >= self.threshold_).astype(int)
        self.validation_metrics_ = {
            "auc_pr": float(average_precision_score(y_val, ensemble_val_calibrated)),
            "auc_roc": float(roc_auc_score(y_val, ensemble_val_calibrated)),
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "threshold": self.threshold_,
            "threshold_precision": threshold_metrics["precision"],
            "threshold_recall": threshold_metrics["recall"],
            "threshold_f1": threshold_metrics["f1"],
        }

        return self

    def _raw_ensemble_probability(self, X):
        X_t = self.preprocessor.transform(X)
        ensemble_prob = np.zeros(len(X_t), dtype=float)
        for name, model in self.fitted_base_models_.items():
            ensemble_prob += self.base_weights_.get(name, 0.0) * model.predict_proba(X_t)[:, 1]
        return ensemble_prob

    def predict_proba(self, X):
        raw_prob = self._raw_ensemble_probability(X)
        calibrated_prob = self.calibrator.transform(raw_prob)
        calibrated_prob = np.asarray(calibrated_prob, dtype=float)
        calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated_prob, calibrated_prob])

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold_).astype(int)

    def summary(self):
        return {
            "base_weights": self.base_weights_,
            "threshold": self.threshold_,
            "validation_metrics": self.validation_metrics_,
        }


def build_meta_baseline_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])
