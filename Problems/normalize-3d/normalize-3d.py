import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.array(v, dtype=float)

    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    elif v.ndim == 2:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return v / norms

    else:
        raise ValueError("Input must have shape (3,) or (N, 3)")