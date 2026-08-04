import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    loss = []

    for y, p in zip(y_true, y_pred):
        p = max(eps, min(1 - eps, p))
        loss.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))

    return loss