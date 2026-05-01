import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from .preprocess import PruningTransformer, PaperFeatureEngineeringTransformer

class PaperStackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_splits=3):
        self.n_splits = n_splits
        self.base_models = [
            ('xgb', XGBClassifier(n_estimators=300, max_depth=9, learning_rate=0.03, scale_pos_weight=25, tree_method='approx', n_jobs=-1, random_state=42)),
            ('lgb', LGBMClassifier(n_estimators=300, num_leaves=127, learning_rate=0.03, scale_pos_weight=25, n_jobs=-1, random_state=42, verbosity=-1)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)),
            ('cat', CatBoostClassifier(iterations=300, depth=8, learning_rate=0.03, scale_pos_weight=25, verbose=False, random_state=42))
        ]
        self.meta_model = LogisticRegression()
        self.classes_ = [0, 1]

    def fit(self, X, y):
        # OOF predictions for Meta-Learner
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X_meta = np.zeros((X.shape[0], len(self.base_models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            
            for model_idx, (name, clf) in enumerate(self.base_models):
                clf.fit(X_fold_train, y_fold_train)
                X_meta[val_idx, model_idx] = clf.predict_proba(X_fold_val)[:, 1]
        
        # Train Meta-model
        self.meta_model.fit(X_meta, y)
        
        # Refit base models on full training data
        for name, clf in self.base_models:
            clf.fit(X, y)
        return self

    def predict_proba(self, X):
        X_meta = np.zeros((X.shape[0], len(self.base_models)))
        for model_idx, (name, clf) in enumerate(self.base_models):
            X_meta[:, model_idx] = clf.predict_proba(X)[:, 1]
        return self.meta_model.predict_proba(X_meta)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def train_model(X_train, y_train):
    print("Building Authoritative Paper Pipeline...")
    pipeline = Pipeline([
        ('pruning', PruningTransformer()),
        ('fe', PaperFeatureEngineeringTransformer()),
        ('model', PaperStackingEnsemble())
    ])
    
    print("Training the Grand Champion Ensemble (this will take time)...")
    pipeline.fit(X_train, y_train)
    return pipeline
