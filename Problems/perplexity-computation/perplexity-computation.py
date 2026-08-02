import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    prob_distributions = np.asarray(prob_distributions)
    actual_tokens = np.asarray(actual_tokens)

    probs = prob_distributions[np.arange(len(actual_tokens)), actual_tokens]
    probs = np.clip(probs, 1e-12, 1.0)

    return np.exp(-np.mean(np.log(probs)))