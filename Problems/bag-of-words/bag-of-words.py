import numpy as np
import collections

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    words_to_count = collections.Counter(tokens)
    ans = []
    for v in vocab:
        ans.append(words_to_count[v])
    ans = np.array(ans, dtype=int)
    return ans