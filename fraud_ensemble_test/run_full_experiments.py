"""Full experiments runner: tuning, multi-model training, imbalance strategies, stacking, calibration, monitoring.

Usage: python run_full_experiments.py --sample 200000
"""
import argparse
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

# ensure local src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features import FeatureEngineeringTransformer
from src.preprocess import PruningTransformer
from src.modeling import FraudEnsembleModel, build_meta_baseline_model
from src.optuna_tuner import tune_xgb
from src.imbalance import apply_smote
from src.stacking import train_stack
from src.calibration import platt_calibration, cv_isotonic_calibration
from src.monitoring import should_retrain

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, classification_report
import joblib


def main(sample: int | None = None):
    root = Path(__file__).resolve().parent.parent
    data_path = root / 'data' / 'merged_raw_train.csv'
    print('Loading data...')
    df = pd.read_csv(data_path)
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=42).sort_values('TransactionDT').reset_index(drop=True)
    else:
        df = df.sort_values('TransactionDT').reset_index(drop=True)

    X = df.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    y = df['isFraud'].astype(int)
    print(f'Using {len(df)} rows (fraud rate {y.mean()*100:.3f}%)')

    # splits
    n = len(df)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    preprocessor = Pipeline([
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('pruning', PruningTransformer(top_k_features=200)),
    ])

    print('Fitting preprocessor...')
    preprocessor.fit(X_train, y_train)
    X_train_feat = preprocessor.transform(X_train)
    X_val_feat = preprocessor.transform(X_val)
    X_test_feat = preprocessor.transform(X_test)
    print('Features:', X_train_feat.shape)

    # Optuna tuning on a subsample
    best_params = None
    try:
        if len(X_train_feat) > 20000:
            sample_idx = np.random.choice(len(X_train_feat), 20000, replace=False)
            print('Running Optuna tuning (20 trials)...')
            best_params = tune_xgb(X_train_feat.iloc[sample_idx], y_train.iloc[sample_idx], n_trials=20)
            print('Optuna best:', best_params)
    except Exception as e:
        print('Optuna tuning skipped:', e)
        best_params = None

    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = neg / pos

    # build candidate models (include CatBoost/LightGBM if available)
    models = {}
    print('Building candidate models...')
    models['xgb_primary'] = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='aucpr', n_jobs=1, random_state=42)
    models['xgb_regularized'] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='aucpr', n_jobs=1, random_state=42)

    # optional LightGBM
    try:
        import lightgbm as lgb
        from sklearn.base import clone
        print('LightGBM available; adding model')
        models['lgb'] = clone(lgb.LGBMClassifier(n_estimators=400, scale_pos_weight=scale_pos_weight))
    except Exception:
        print('LightGBM not available; skipping')

    # optional CatBoost
    try:
        from catboost import CatBoostClassifier
        from sklearn.base import clone
        print('CatBoost available; adding model')
        models['catboost'] = CatBoostClassifier(iterations=400, verbose=0, auto_class_weights='Balanced')
    except Exception:
        print('CatBoost not available; skipping')

    models['logistic_baseline'] = build_meta_baseline_model()

    # imbalance: try SMOTE
    X_train_used, y_train_used = X_train_feat, y_train
    try:
        X_train_used, y_train_used = apply_smote(X_train_feat, y_train)
        print('SMOTE applied: result size', len(X_train_used))
    except Exception:
        print('SMOTE not applied or not available')

    # stacking OOF
    print('Training stacking OOF...')
    meta_X, meta_y, fitted = train_stack(models, X_train_used, y_train_used, n_folds=5)
    print('Meta shape', meta_X.shape)
    meta_learner = LogisticRegression(max_iter=2000)
    meta_learner.fit(meta_X, meta_y)

    # build ensemble wrapper using fitted base models
    ensemble = FraudEnsembleModel(preprocessor=preprocessor, base_models=fitted)
    # set meta learner if model supports it
    try:
        ensemble.meta_learner_ = meta_learner
    except Exception:
        pass

    # calibrate
    try:
        val_probs = np.column_stack([m.predict_proba(X_val_feat)[:, 1] for m in fitted.values()])
        meta_prob = meta_learner.predict_proba(val_probs)[:, 1]
        calib = platt_calibration(meta_prob, y_val.values)
        ensemble.calibrator_ = calib
        print('Calibration fitted')
    except Exception as e:
        print('Calibration skipped:', e)

    # evaluate
    test_probs = None
    try:
        test_probas = np.column_stack([m.predict_proba(X_test_feat)[:, 1] for m in fitted.values()])
        test_probs = meta_learner.predict_proba(test_probas)[:, 1]
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(y_test, test_probs)
        f1 = (2 * p * r) / (p + r + 1e-9)
        best_idx = f1.argmax()
        threshold = float(t[best_idx] if best_idx < len(t) else 0.5)
        preds = (test_probs >= threshold).astype(int)
        pr_auc = average_precision_score(y_test, test_probs)
        roc_auc = roc_auc_score(y_test, test_probs)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        print('\nRESULTS:')
        print('PR-AUC', pr_auc, 'ROC-AUC', roc_auc, 'Precision', precision, 'Recall', recall, 'threshold', threshold)
    except Exception as e:
        print('Evaluation failed:', e)

    # save
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = Path(__file__).resolve().parent / 'models'
    out.mkdir(exist_ok=True)
    joblib.dump({'fitted_bases': fitted, 'meta_learner': meta_learner}, out / f'experiment_{ts}.pkl')
    print('Saved experiment to', out / f'experiment_{ts}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=200000)
    args = parser.parse_args()
    main(sample=args.sample)
