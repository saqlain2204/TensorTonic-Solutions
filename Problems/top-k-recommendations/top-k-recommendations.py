def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    rated_indices = set(rated_indices)
    unrated = [i for i in range(len(scores)) if i not in rated_indices]
    unrated.sort(key=lambda i: scores[i], reverse=True)

    return unrated[:k]
