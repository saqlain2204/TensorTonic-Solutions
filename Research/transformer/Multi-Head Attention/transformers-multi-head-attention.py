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
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projections
    Q = np.dot(Q, W_q)
    K = np.dot(K, W_k)
    V = np.dot(V, W_v)

    # Split into heads
    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    heads = np.matmul(weights, V)

    # Concatenate heads
    heads = heads.transpose(0, 2, 1, 3)
    heads = heads.reshape(batch, seq_len, d_model)

    # Final projection
    return np.dot(heads, W_o)
    