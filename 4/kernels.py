import numpy as np


def rbf(
        X: np.ndarray,
        w: np.ndarray,
        gamma: float
) -> np.ndarray:
    return np.exp(
        -gamma * np.sum((X - w) ** 2, axis=1)
    )
