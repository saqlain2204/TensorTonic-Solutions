import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g, dtype=float)
    clipped = g.copy()

    norm = np.linalg.norm(g)

    if norm == 0 or max_norm <= 0 or norm <= max_norm:
        return clipped

    return g * (max_norm / norm)