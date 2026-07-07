import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    n = len(x)

    mean = np.mean(x)

    sample_variance = np.sum((x - mean) ** 2) / (n - 1)
    sample_std = np.sqrt(sample_variance)

    return sample_variance, sample_std


    