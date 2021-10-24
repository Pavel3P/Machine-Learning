import numpy as np
from scipy.optimize import minimize
from misc.activation_functions import relu


class SVM:
    l: float
    w: np.ndarray
    w0: float

    def __init__(self,
                 l: float = 1.0
                 ) -> None:
        self.l = l

    def train(self,
              X: np.ndarray,
              Y: np.ndarray
              ) -> None:
        params: np.ndarray = minimize(
            lambda W: np.mean(
                np.vectorize(relu)(1 - Y * (X @ W[:-1] + W[-1]))
            ) + np.linalg.norm(W) ** 2 * self.l / 2,
            x0=np.random.rand(X.shape[1] + 1)
        ).x
        self.w = params[:-1]
        self.w0 = params[-1]

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        return np.sign(X @ self.w + self.w0)
