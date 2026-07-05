import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def attention(q, k, v):
    d_k = k.shape[-1]

    scores = q @ k.T
    scores = scores / np.sqrt(d_k)

    weights = softmax(scores)

    return weights @ v
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    heads = []

    for i in range(num_heads):
        q_i = Q @ W_q[i]
        k_i = K @ W_k[i]
        v_i = V @ W_v[i]

        head_i = attention(q_i, k_i, v_i)
        heads.append(head_i)

    heads = np.concatenate(heads, axis=-1)
    return heads @ W_o
    