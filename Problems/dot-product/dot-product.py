import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    return np.dot(x, y)