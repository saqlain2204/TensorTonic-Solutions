import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len).reshape(seq_len, 1)
    i = np.arange(d_model).reshape(1, d_model)

    angles = pos / (base ** (2*(i // 2) / d_model))
    pe = np.where(i%2==0, np.sin(angles), np.cos(angles))

    return pe
    