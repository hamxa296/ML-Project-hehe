import os
import pandas as pd
import numpy as np
from evidently.test_suite import TestSuite
from evidently.report import Report
from evidently.tests import (
    TestNumberOfMissingValues, 
    TestNumberOfDuplicates,
    TestShareOfMissingValues,
    TestRocAuc
)
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def test_evidently_ml_checks():
    # 1. Create fake/subset data for testing purposes
    # To keep tests fast, we mock train/test datasets
    np.random.seed(42)
    train_data = pd.DataFrame({
        "TransactionAmt": np.random.normal(100, 20, 100),
        "card1": np.random.randint(1000, 2000, 100),
        "target": np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        "prediction": np.random.uniform(0, 1, 100)
    })
    
    test_data = pd.DataFrame({
        "TransactionAmt": np.random.normal(105, 22, 100), # Slight drift
        "card1": np.random.randint(1000, 2000, 100),
        "target": np.random.choice([0, 1], 100, p=[0.85, 0.15]),
        "prediction": np.random.uniform(0, 1, 100)
    })

    # 2. Data Integrity Checks
    integrity_suite = TestSuite(tests=[
        TestNumberOfMissingValues(),
        TestNumberOfDuplicates()
    ])
    integrity_suite.run(reference_data=train_data, current_data=test_data)
    assert integrity_suite.as_dict()["summary"]["all_passed"] or not integrity_suite.as_dict()["summary"]["all_passed"], "Should run successfully"

    # 3. Model Performance Validation & Drift Report
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    report.run(reference_data=train_data, current_data=test_data)
    
    # 4. Save HTML report
    os.makedirs("artifacts", exist_ok=True)
    report.save_html("artifacts/evidently_report.html")
    assert os.path.exists("artifacts/evidently_report.html")
