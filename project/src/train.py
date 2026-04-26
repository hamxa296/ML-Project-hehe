from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.preprocess import PruningTransformer
from src.features import FeatureEngineeringTransformer, ClusteringTransformer

def train_model(X_train, y_train):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=500, 
        max_depth=12, 
        learning_rate=0.02, 
        scale_pos_weight=ratio, 
        tree_method='approx', 
        subsample=0.8, 
        colsample_bytree=0.8, 
        n_jobs=-1, 
        random_state=42
    )

    pipeline = Pipeline([
        ('prune', PruningTransformer()),
        ('fe', FeatureEngineeringTransformer()),
        ('clustering', ClusteringTransformer(n_clusters=5)),
        ('model', xgb)
    ])

    print("Training the Unified Paper-Exact Pipeline...")
    pipeline.fit(X_train, y_train)
    return pipeline
