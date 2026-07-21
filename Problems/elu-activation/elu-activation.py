import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    x = np.array(x, dtype = float)
    return list(np.where(
        x > 0,
        x,
        alpha * (np.exp(x) - 1)
    ))