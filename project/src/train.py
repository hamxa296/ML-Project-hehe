import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

from .preprocess import PruningTransformer
from .features import FeatureEngineeringTransformer, ClusteringTransformer

def train_model(X_train, y_train):
    print("Building Integrated ML Pipeline...")
    
    # Calculate ratio of Safe/Fraud transactions
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    
    pipeline = Pipeline([
        ('prune', PruningTransformer()),
        ('fe', FeatureEngineeringTransformer()),
        ('clustering', ClusteringTransformer(n_clusters=5)),
        ('model', XGBClassifier(
            n_estimators=500, 
            max_depth=12, 
            learning_rate=0.02, 
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight, 
            tree_method='approx', 
            n_jobs=-1, 
            random_state=42
        ))
    ])
    
    print("Training XGBoost + Clustering (this will take time)...")
    pipeline.fit(X_train, y_train)
    return pipeline
