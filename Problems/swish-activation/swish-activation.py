import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.array(x, dtype=float)
    return x*sigmoid(x)
    