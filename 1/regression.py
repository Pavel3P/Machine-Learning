import numpy as np
from scipy.optimize import minimize


class LinearRegression:
    def __init__(self, l1: float = 0, l2: float = 0) -> None:
        self.l1: float = l1
        self.l2: float = l2
        self.w: np.ndarray = None

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = np.asarray(X)
        self.w = minimize(
            lambda w: self.__error(w, np.concatenate((np.ones((X.shape[0], 1)), X), axis=1), Y),
            x0=np.zeros(X.shape[1] + 1)
        ).x

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) @ self.w

    def MSE(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = np.asarray(X)
        return np.square(
            np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) @ self.w - Y
        ).mean()

    def __error(self, w: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
        return np.mean(np.square(X @ w - Y)) \
               + self.l1 * np.sum(np.abs(w)) \
               + self.l2 * np.sum(np.square(w))
