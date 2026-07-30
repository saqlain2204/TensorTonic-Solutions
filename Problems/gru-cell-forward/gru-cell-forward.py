import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    x, x_was_1d = _as2d(x, Wz.shape[0])
    h_prev, h_was_1d = _as2d(h_prev, Uz.shape[0])

    z_t = _sigmoid(x @ Wz + h_prev @ Uz + bz)
    r_t = _sigmoid(x @ Wr + h_prev @ Ur + br)
    h_hat = np.tanh(x @ Wh + (r_t * h_prev) @ Uh + bh)

    h_t = (1 - z_t) * h_prev + z_t * h_hat

    if x_was_1d:
        h_t = h_t.squeeze(0)

    return h_t

    