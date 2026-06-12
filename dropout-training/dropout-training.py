import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    random_values = rng.random(x.shape)

    dropout_pattern = np.where(
        random_values < (1 - p),
        1 / (1 - p),
        0
    )

    output = x * dropout_pattern

    return output, dropout_pattern