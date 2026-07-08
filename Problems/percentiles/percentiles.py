import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.array(x, dtype=float)
    q = np.array(q, dtype=float)

    return np.percentile(x, q, method='linear')