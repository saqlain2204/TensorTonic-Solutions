import numpy as np


def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0)

    if X.ndim != 2:
        return None

    N, D = X.shape

    if N < 2:
        return None

    x_centered = X - mean

    cov_matrix = (x_centered.T @ x_centered) / (N - 1)

    return cov_matrix
