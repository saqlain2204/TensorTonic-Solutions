import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    abs_a = np.linalg.norm(a)
    abs_b = np.linalg.norm(b)

    if not abs_a or not abs_b:
        return 0.0

    return np.dot(a, b)/(abs_a*abs_b)