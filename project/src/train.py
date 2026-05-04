import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

from .preprocess import PruningTransformer
from .features import FeatureEngineeringTransformer, ClusteringTransformer, PCATransformer, AssociationTransformer

def train_model(X_train, y_train):
    print("Building Augmented Integrated ML Pipeline...")
    
    # Calculate ratio of Safe/Fraud transactions
    # Calculate ratio of Safe/Fraud transactions
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    # Use dampened scale weight (sqrt) to prevent extreme precision loss
    dampened_weight = np.sqrt(scale_weight) if scale_weight > 1 else 1
    
    pipeline = Pipeline([
        ('prune', PruningTransformer()),
        ('fe', FeatureEngineeringTransformer()),
        ('clustering', ClusteringTransformer(n_clusters=5)),
        ('pca', PCATransformer(n_components=5)),
        ('association', AssociationTransformer(min_support=0.01, min_confidence=0.05, top_n=10)),
        ('model', XGBClassifier(
            n_estimators=600, 
            max_depth=10, 
            learning_rate=0.02, 
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=dampened_weight, 
            min_child_weight=10,
            reg_lambda=2.0,
            tree_method='approx', 
            n_jobs=-1, 
            random_state=42
        ))
    ])
    
    print("Training XGBoost + Clustering (this will take time)...")
    pipeline.fit(X_train, y_train)
    return pipeline
