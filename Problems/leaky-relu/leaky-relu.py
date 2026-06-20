import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.array(x, dtype=float)
    x = np.where(x >= 0, x, x*alpha)

    return x
    