import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred, y_true = np.array(y_pred, dtype=float), np.array(y_true, dtype=float)
    N = len(y_pred)

    return np.mean(np.square(y_pred-y_true))
