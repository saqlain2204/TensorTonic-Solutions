import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.array(x, dtype=float)
    mean = np.mean(x)
    median = np.median(x)

    c = Counter(x)
    mode = 0
    max = 0
    for k, v in c.items():
        if v > max:
            max = v
            mode = k

    return mean, median, mode