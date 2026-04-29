import numpy as np


def kl_divergence(p, q, eps=1e-9):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    return np.sum(p * np.log(p / q))


def simple_distribution_drift(prev_scores, new_scores, n_bins=50):
    # compute histogram KL divergence
    prev_hist, edges = np.histogram(prev_scores, bins=n_bins, range=(0.0, 1.0), density=True)
    new_hist, _ = np.histogram(new_scores, bins=edges, density=True)
    return kl_divergence(prev_hist, new_hist)


def should_retrain(prev_scores, new_scores, threshold=0.1):
    d = simple_distribution_drift(prev_scores, new_scores)
    return d > threshold, d
