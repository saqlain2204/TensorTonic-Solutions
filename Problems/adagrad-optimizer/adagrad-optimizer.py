import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    G_t = G + np.pow(g, 2)

    wt = w - (lr/np.sqrt(G_t + eps))*g

    return wt, G_t