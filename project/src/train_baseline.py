import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

def train_baseline_model(X_train, y_train):
    """
    Trains a simple Logistic Regression baseline with NO advanced 
    feature engineering or clustering.
    """
    print(">>> [Baseline] Training Logistic Regression Baseline...")
    
    # We only take numeric columns for the simple baseline
    X_train_num = X_train.select_dtypes(include=['number']).fillna(0)
    
    # Simple Pipeline: Scale -> Logistic Regression
    baseline_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    baseline_pipe.fit(X_train_num, y_train)
    return baseline_pipe

def evaluate_baseline(pipeline, X_test, y_test):
    X_test_num = X_test.select_dtypes(include=['number']).fillna(0)
    probs = pipeline.predict_proba(X_test_num)[:, 1]
    
    auc_pr  = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)
    
    return {
        "auc_pr":  auc_pr,
        "auc_roc": auc_roc
    }
