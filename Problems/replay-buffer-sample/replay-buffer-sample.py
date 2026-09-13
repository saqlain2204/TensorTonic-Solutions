import numpy as np

def replay_buffer_sample(buffer: list, batch_size: int, seed: int) -> list:
    """
    Returns a deterministic sample of transitions.
    """
    # Write code here
    np.random.seed(seed)
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    indices.sort()
    return [buffer[i] for i in indices]
