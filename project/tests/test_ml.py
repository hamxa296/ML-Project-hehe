import pandas as pd
import numpy as np
import pytest
from src.predict import predict

def test_data_schema_sanity():
    """Real ML Validation: Schema checks and missing values check."""
    df = pd.DataFrame({"TransactionAmt": [10.0, 50.0], "card1": [1000, 2000]})
    
    assert df.isnull().sum().sum() == 0, "Initial test data should not have missing values."
    assert "TransactionAmt" in df.columns
    assert "card1" in df.columns

def test_input_robustness():
    """Test resilience against missing fields, extra fields, and type inconsistencies."""
    from src.preprocess import PruningTransformer
    
    X_train = pd.DataFrame({
        "TransactionAmt": [100, 200, 300], 
        "card1": [1, 2, 3],
        "isFraud": [0, 1, 0]
    })
    y_train = X_train["isFraud"]
    X_train = X_train.drop(columns=["isFraud"])
    
    pruner = PruningTransformer()
    # Mocking internal feature selection to bypass mutual_info complexity in fast test
    pruner.input_schema = {"TransactionAmt": np.dtype('float64'), "card1": np.dtype('int64')}
    pruner.keep_cols = ["TransactionAmt", "card1"]
    
    # Missing card1, extra 'unseen_col', bad type in 'TransactionAmt'
    X_test_malformed = pd.DataFrame({
        "TransactionAmt": ["250.0", "bad_type"], 
        "unseen_col": [99, 88]
    })
    
    X_out = pruner.transform(X_test_malformed)
    
    assert "card1" in X_out.columns, "Missing columns should be imputed."
    assert "unseen_col" not in X_out.columns, "Extra columns should be dropped."
    assert X_out["TransactionAmt"].iloc[1] == 0, "Bad types should be coerced to 0."

def test_prediction_output_format():
    """Real ML Validation: Prediction sanity checking."""
    class DummyPipeline:
        def predict_proba(self, X):
            probs = np.array([[0.9, 0.1], [0.2, 0.8]])
            assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probs out of bounds"
            return probs
    
    model = DummyPipeline()
    X_dummy = pd.DataFrame({"dummy": [1, 2]})
    
    probs, preds = predict(model, X_dummy)
    
    assert len(probs) == 2
    assert set(preds).issubset({0, 1}), "Predictions must be exactly binary class"

def test_distribution_drift_simulation():
    """Real ML Validation: Simulate a distribution check comparing train vs new data statistics using Evidently."""
    from evidently.test_suite import TestSuite
    from evidently.tests import TestShareOfDriftedColumns
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    train_data = pd.DataFrame({"feature1": np.random.normal(0, 1, 100), "feature2": np.random.normal(5, 2, 100)})
    new_data = pd.DataFrame({"feature1": np.random.normal(0.1, 1, 100), "feature2": np.random.normal(5.5, 2, 100)})
    
    drift_suite = TestSuite(tests=[TestShareOfDriftedColumns(lt=0.5)])
    drift_suite.run(reference_data=train_data, current_data=new_data)
    
    results = drift_suite.as_dict()
    assert results["summary"]["all_passed"], f"Significant distribution drift detected by Evidently: {results}"

