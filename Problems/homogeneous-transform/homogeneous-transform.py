import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    T = np.array(T)
    points = np.array(points)

    if points.ndim == 1:
        return T[:3, :3] @ points + T[:3, 3]

    return points @ T[:3, :3].T + T[:3, 3]
