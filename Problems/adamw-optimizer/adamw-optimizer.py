import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    mt = np.dot(beta1, m) + np.dot((1 - beta1), grad)
    vt = np.dot(beta2, v) + np.dot((1 - beta2), np.pow(grad, 2))

    wt = w - lr*(np.dot(weight_decay, w)) - lr*(mt/(np.sqrt(vt) + eps))

    return wt, mt, vt