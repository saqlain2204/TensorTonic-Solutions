def discount_returns(rewards: list, gamma: float) -> list:
    """
    Returns the discounted return at every timestep.
    """
    # Write code here
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns
