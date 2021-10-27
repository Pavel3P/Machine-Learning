import numpy as np
from scipy.optimize import minimize


class SVM:
    w: np.ndarray
    w0: float

    def __init__(self,
                 l: float = 1.0,
                 kernel: callable = lambda X, w: X @ w
                 ) -> None:
        self.l: float = l
        self.kernel: callable = kernel

    def train(self,
              X: np.ndarray,
              Y: np.ndarray
              ) -> None:
        to_minimize: callable = lambda W: np.mean(
            # Hinge loss
            np.clip(1 - Y * (self.kernel(X, W[:-1]) + W[-1]), 0, np.inf)
        ) + self.l * np.linalg.norm(W) ** 2
        params: np.ndarray = minimize(
            to_minimize,
            x0=np.random.rand(X.shape[1] + 1)
        ).x

        self.w = params[:-1]
        self.w0 = params[-1]

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        return np.sign(self.kernel(X, self.w) + self.w0)
