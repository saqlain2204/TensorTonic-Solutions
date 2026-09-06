def sarsa_update(q_table: list, state: int, action: int, reward: float, next_state: int, next_action: int, alpha: float, gamma: float) -> list:
    """
    Returns a copied Q-table after one SARSA update.
    """
    # Write code here
    
    new_q_table = [row[:] for row in q_table]
    delta = (reward + gamma * new_q_table[next_state][next_action]- new_q_table[state][action])
    new_q_table[state][action] += alpha * delta

    return new_q_table


