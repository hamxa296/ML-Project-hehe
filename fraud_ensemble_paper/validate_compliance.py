#!/usr/bin/env python3
"""
Compliance Validation Script for IEEE-CIS Fraud Detection Paper Replication

This script systematically validates that the implementation matches the paper's
specifications and reports metrics against Table 4 of the paper.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_ensemble_paper.src.paper_pipeline import train_paper_style


# Paper-Reported Baseline Metrics (Table 4)
PAPER_METRICS = {
    "Proposed Stacking": {"auc_roc": 0.918, "auc_pr": 0.891, "f1": 0.856, "bal_acc": 0.810, "g_mean": 0.824},
    "XGBoost": {"auc_roc": 0.887, "auc_pr": 0.834, "f1": 0.802, "bal_acc": 0.756, "g_mean": 0.771},
    "LightGBM": {"auc_roc": 0.882, "auc_pr": 0.821, "f1": 0.791, "bal_acc": 0.741, "g_mean": 0.756},
    "Random Forest": {"auc_roc": 0.865, "auc_pr": 0.796, "f1": 0.768, "bal_acc": 0.712, "g_mean": 0.724},
    "Weighted Voting": {"auc_roc": 0.901, "auc_pr": 0.861, "f1": 0.828, "bal_acc": 0.784, "g_mean": 0.801},
    "Simple Voting": {"auc_roc": 0.885, "auc_pr": 0.819, "f1": 0.784, "bal_acc": 0.734, "g_mean": 0.748},
    "CatBoost": {"auc_roc": 0.890, "auc_pr": 0.837, "f1": 0.805, "bal_acc": 0.761, "g_mean": 0.776},
    "Logistic Regression": {"auc_roc": 0.812, "auc_pr": 0.721, "f1": 0.651, "bal_acc": 0.593, "g_mean": 0.603},
}


class ComplianceValidator:
    """Validates paper compliance and compares metrics."""

    def __init__(self, data_path: str = "data/merged_raw_train.csv", sample_size: int = None):
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        self.results = {}
        self.errors = []
        self.warnings = []

    def log_error(self, message: str) -> None:
        """Log an error."""
        print(f"❌ ERROR: {message}")
        self.errors.append(message)

    def log_warning(self, message: str) -> None:
        """Log a warning."""
        print(f"⚠️  WARNING: {message}")
        self.warnings.append(message)

    def log_info(self, message: str) -> None:
        """Log an info message."""
        print(f"ℹ️  INFO: {message}")

    def log_success(self, message: str) -> None:
        """Log a success message."""
        print(f"✅ SUCCESS: {message}")

    def check_data_loading(self) -> bool:
        """Check if data can be loaded."""
        print("\n" + "=" * 80)
        print("PHASE 1: DATA LOADING CHECK")
        print("=" * 80)

        if not self.data_path.exists():
            self.log_error(f"Data file not found: {self.data_path}")
            return False

        try:
            df = pd.read_csv(self.data_path, nrows=self.sample_size)
            self.log_success(f"Data loaded: {len(df):,} rows x {len(df.columns)} columns")

            # Check required columns
            required_cols = ["isFraud", "TransactionDT", "TransactionID"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                self.log_error(f"Missing required columns: {missing_cols}")
                return False

            # Check feature count (should be 431 baseline before engineering)
            feature_cols = [c for c in df.columns if c not in required_cols]
            expected_baseline = 431
            if len(feature_cols) != expected_baseline:
                self.log_warning(
                    f"Baseline feature count mismatch: got {len(feature_cols)}, expected ~{expected_baseline}"
                )

            # Check fraud distribution
            fraud_rate = df["isFraud"].mean()
            self.log_info(f"Fraud rate: {fraud_rate:.4f} (expected ~0.0348)")

            self.results["data_loaded"] = True
            self.results["n_rows"] = len(df)
            self.results["n_features_baseline"] = len(feature_cols)
            self.results["fraud_rate"] = fraud_rate

            return True
        except Exception as e:
            self.log_error(f"Data loading failed: {e}")
            return False

    def check_feature_engineering(self) -> bool:
        """Check feature engineering pipeline."""
        print("\n" + "=" * 80)
        print("PHASE 2: FEATURE ENGINEERING CHECK")
        print("=" * 80)

        try:
            from fraud_ensemble_paper.src.paper_pipeline import train_paper_style

            result = train_paper_style(
                data_path=str(self.data_path), sample=self.sample_size, n_folds=2
            )

            n_features = result.get("feature_count", 0)
            self.log_info(f"Features after preprocessing: {n_features}")

            expected_features = 167
            if n_features != expected_features:
                gap = n_features - expected_features
                pct = (gap / expected_features) * 100
                self.log_warning(
                    f"Feature count mismatch: got {n_features}, expected {expected_features} (gap: +{gap} / +{pct:.1f}%)"
                )
            else:
                self.log_success(f"Feature count matches paper specification: {n_features}")

            self.results["feature_engineering"] = result
            self.results["n_features_final"] = n_features
            return True
        except Exception as e:
            self.log_error(f"Feature engineering failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def compute_g_mean(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute G-Mean (geometric mean of sensitivity and specificity)."""
        sensitivity = recall_score(y_true, y_pred)
        specificity = recall_score(y_true, 1 - y_pred)
        g_mean = np.sqrt(sensitivity * specificity)
        return g_mean

    def validate_metrics(self) -> bool:
        """Validate metrics against paper."""
        print("\n" + "=" * 80)
        print("PHASE 3: METRICS VALIDATION")
        print("=" * 80)

        if "feature_engineering" not in self.results:
            self.log_error("Cannot validate metrics: feature engineering not completed")
            return False

        result = self.results["feature_engineering"]

        test_metrics = result.get("test_metrics", {})
        metrics_to_check = {
            "OOF PR-AUC": result.get("oof_pr_auc"),
            "Stacked PR-AUC": test_metrics.get("stacked_pr_auc"),
            "Weighted Vote PR-AUC": test_metrics.get("weighted_vote_pr_auc"),
            "Stacked ROC-AUC": test_metrics.get("stacked_roc_auc"),
            "Weighted Vote ROC-AUC": test_metrics.get("weighted_vote_roc_auc"),
        }

        print("\nCURRENT IMPLEMENTATION RESULTS:")
        print("-" * 80)
        for name, current_value in metrics_to_check.items():
            if isinstance(current_value, (int, float, np.floating)):
                print(f"  {name:.<30} {float(current_value):.6f}")
            else:
                print(f"  {name:.<30} N/A")

        print("\nPAPER TARGET METRICS (Table 4):")
        print("-" * 80)
        for method, metrics in PAPER_METRICS.items():
            if method in ["Proposed Stacking", "Weighted Voting"]:
                print(
                    f"  {method:.<30} PR-AUC: {metrics['auc_pr']:.6f} | "
                    f"ROC-AUC: {metrics['auc_roc']:.6f} | F1: {metrics['f1']:.6f}"
                )

        print("\nMETRIC GAPS:")
        print("-" * 80)

        stacking_paper_pr_auc = PAPER_METRICS["Proposed Stacking"]["auc_pr"]
        stacking_current_pr_auc = test_metrics.get("stacked_pr_auc")
        if not isinstance(stacking_current_pr_auc, (int, float, np.floating)):
            stacking_current_pr_auc = 0.0
        gap = stacking_paper_pr_auc - stacking_current_pr_auc
        pct_gap = (gap / stacking_paper_pr_auc) * 100

        print(f"  Stacking PR-AUC Gap:  {gap:+.6f} ({pct_gap:+.1f}%)")
        print(f"    Paper Target: {stacking_paper_pr_auc:.6f}")
        print(f"    Current:      {stacking_current_pr_auc:.6f}")

        if abs(gap) > 0.05:
            self.log_warning(f"Large metric gap detected: {abs(gap):.6f}")

        self.results["metrics_validation"] = {
            "stacking_pr_auc_gap": gap,
            "stacking_pr_auc_gap_pct": pct_gap,
        }

        return True

    def generate_report(self) -> str:
        """Generate final compliance report."""
        print("\n" + "=" * 80)
        print("COMPLIANCE REPORT SUMMARY")
        print("=" * 80)

        # Safely format numeric values that may be missing
        stacked_actual = self.results.get('feature_engineering', {}).get('test_metrics', {}).get('stacked_pr_auc', None)
        stacked_actual_fmt = f"{stacked_actual:.6f}" if isinstance(stacked_actual, (int, float)) else str(stacked_actual)

        summary = f"""
EXECUTION SUMMARY
{'-' * 80}
Data Path:            {self.data_path}
Sample Size:          {self.sample_size if self.sample_size else 'FULL DATASET'}
Rows Processed:       {self.results.get('n_rows', 'N/A'):,}
Baseline Features:    {self.results.get('n_features_baseline', 'N/A')}
Final Features:       {self.results.get('n_features_final', 'N/A')}
Fraud Rate:           {self.results.get('fraud_rate', 'N/A'):.4f}

COMPLIANCE STATUS
{'-' * 80}
✅ Data Loading:                {self.results.get('data_loaded', False)}
✅ Feature Engineering:         {self.results.get('n_features_final') is not None}
⚠️  Metric Gap Analysis:        See details below

KEY FINDINGS
{'-' * 80}
Feature Count Target: 167 (Paper Table 3)
Feature Count Actual: {self.results.get('n_features_final', 'Unknown')}
Feature Match:        {'✅ MATCH' if self.results.get('n_features_final') == 167 else '❌ MISMATCH'}

Stacking PR-AUC Target: {PAPER_METRICS['Proposed Stacking']['auc_pr']:.6f} (Paper Table 4)
Stacking PR-AUC Actual: {stacked_actual_fmt}
Metric Gap:             {self.results.get('metrics_validation', {}).get('stacking_pr_auc_gap', 0):+.6f}
Gap Percentage:         {self.results.get('metrics_validation', {}).get('stacking_pr_auc_gap_pct', 0):+.1f}%

ERRORS & WARNINGS ({len(self.errors)} errors, {len(self.warnings)} warnings)
{'-' * 80}
"""

        for error in self.errors:
            summary += f"❌ {error}\n"
        for warning in self.warnings:
            summary += f"⚠️  {warning}\n"

        summary += f"""
NEXT STEPS
{'-' * 80}
1. Verify feature pruning pipeline produces exactly 167 features
2. Add LightGBM and CatBoost to base learner ensemble
3. Run full dataset (590k samples) evaluation
4. Compare all metrics from Table 4
5. Document any persistent gaps with root cause analysis
"""

        return summary

    def run_validation(self) -> bool:
        """Run full compliance validation."""
        print("\n" + "🔍" * 40)
        print("IEEE-CIS FRAUD DETECTION - COMPLIANCE VALIDATION")
        print("🔍" * 40)

        # Phase 1: Data Loading
        if not self.check_data_loading():
            print("\n❌ Validation failed at data loading phase")
            return False

        # Phase 2: Feature Engineering
        if not self.check_feature_engineering():
            print("\n❌ Validation failed at feature engineering phase")
            return False

        # Phase 3: Metrics Validation
        if not self.validate_metrics():
            print("\n❌ Validation failed at metrics validation phase")
            return False

        # Generate report
        report = self.generate_report()
        print(report)

        # Save report
        report_path = Path(__file__).parent / "COMPLIANCE_REPORT.txt"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"\n📄 Report saved to: {report_path}")

        return len(self.errors) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate IEEE-CIS fraud detection paper compliance")
    parser.add_argument("--data", default="data/merged_raw_train.csv", help="Path to data CSV")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (None = full dataset)")
    parser.add_argument("--full", action="store_true", help="Run on full dataset (ignore sample size)")

    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    validator = ComplianceValidator(data_path=args.data, sample_size=sample_size)
    success = validator.run_validation()

    sys.exit(0 if success else 1)
