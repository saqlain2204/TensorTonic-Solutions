import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)
    total = np.zeros(n_states)
    count = np.zeros(n_states)

    for episode in episodes:
        visited = set()

        for t, (state, reward) in enumerate(episode):
            if state in visited:
                continue

            visited.add(state)

            G = 0
            for k in range(t, len(episode)):
                G += (gamma ** (k - t)) * episode[k][1]

            total[state] += G
            count[state] += 1

    for state in range(n_states):
        if count[state] > 0:
            V[state] = total[state] / count[state]

    return V
