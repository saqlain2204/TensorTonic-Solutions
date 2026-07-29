import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here

    layer_1 = x @ W1 + b1
    layer_2 = np.maximum(0, layer_1) @ W2 + b2

    return layer_2