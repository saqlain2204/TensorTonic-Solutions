import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w_look = w - np.dot(momentum, v)
    v = np.dot(momentum, v) + np.dot(lr, grad, w_look)
    w = w - v

    return w, v