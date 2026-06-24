import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here

    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    
    loss = np.maximum(0, margin - y_true*y_score)

    if reduction == "mean":
        return np.mean(loss)
    if reduction == "sum":
        return np.sum(loss)