import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    states = np.array(states)
    V = np.array(V)
    returns = np.zeros(len(states))

    G = 0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        returns[i] = G

    advantage = returns - V[states]

    return advantage
