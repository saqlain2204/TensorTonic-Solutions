import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    v_dim = v.ndim

    if v_dim != 1:
        return np.sum(v**2, axis = 1)**0.5

    else:
        return np.sum(v**2)**0.5
