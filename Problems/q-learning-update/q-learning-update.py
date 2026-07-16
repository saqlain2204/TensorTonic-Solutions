import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    # Write code here
    Q = np.array(Q, dtype=float)
    Q[s][a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
    return Q