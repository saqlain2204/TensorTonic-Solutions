def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    if len(recommendations) != len(ground_truth):
        raise ValueError("recommendations and ground_truth must have the same length")

    if not recommendations:
        return 0.0

    hits = 0

    for recs, truth in zip(recommendations, ground_truth):
        if set(recs[:k]) & set(truth):
            hits += 1

    return hits / len(recommendations)
