from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.base import clone


def train_stack(base_models: dict, X, y, n_folds: int = 5):
    """Train base models in out-of-fold fashion and return meta training matrix and fitted base models.

    base_models: dict name->unfitted estimator
    Returns: (meta_X, meta_y, fitted_base_models)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    meta_X = np.zeros((len(X), len(base_models)))
    meta_y = np.array(y)
    fitted_models = {name: [] for name in base_models}
    for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y.iloc[tr]
        for col_idx, (name, model) in enumerate(base_models.items()):
            m = clone(model)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_va)[:, 1]
            meta_X[va, col_idx] = p
            fitted_models[name].append(m)
    # for inference, keep one ensemble of each base model by refitting on full set
    refit_models = {}
    for name, model in base_models.items():
        m = clone(model)
        m.fit(X, y)
        refit_models[name] = m
    return meta_X, meta_y, refit_models
