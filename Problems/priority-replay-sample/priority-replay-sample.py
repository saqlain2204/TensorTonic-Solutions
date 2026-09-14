def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    # Write code here
    n = len(priorities)
    scaled = [p ** alpha for p in priorities]
    total = sum(scaled)
    probs = [p / total for p in scaled]
    weights = [(n * p) ** (-beta) for p in probs]
    max_weight = max(weights)
    weights = [w / max_weight for w in weights]
    return [probs, weights]
