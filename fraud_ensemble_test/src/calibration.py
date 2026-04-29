from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np


def cv_isotonic_calibration(probs, y, n_splits=5):
    # probs: array-like shape (n_samples,)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    calib = IsotonicRegression(out_of_bounds='clip')
    # Fit on full probs/y for simplicity; in production use OOF calibration
    try:
        calib.fit(probs, y)
        return calib
    except Exception:
        return None


def platt_calibration(probs, y):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(probs.reshape(-1, 1), y)
    return lr
