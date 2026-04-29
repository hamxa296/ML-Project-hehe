import warnings

try:
    import optuna
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.metrics import average_precision_score
except Exception:
    optuna = None


def tune_xgb(X, y, n_trials: int = 20, n_splits: int = 3, random_state: int = 42):
    if optuna is None:
        warnings.warn("optuna or xgboost not available; skipping tuning")
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        for tr, va in skf.split(X, y):
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric="aucpr", n_jobs=1)
            model.fit(X.iloc[tr], y.iloc[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            scores.append(average_precision_score(y.iloc[va], p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
