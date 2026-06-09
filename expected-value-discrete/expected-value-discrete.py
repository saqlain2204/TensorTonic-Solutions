import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)
    
    if np.sum(p) != 1:
        raise ValueError("Probs dont sum to 1")

    assert x.shape == p.shape

    return np.dot(x, p)
