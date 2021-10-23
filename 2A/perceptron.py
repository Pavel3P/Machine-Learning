import numpy as np


class Perceptron:
    """
    Implementation of perceptron with
    Rosenblatt's training algorithm
    for binary classification.
    """

    def __init__(self
                 ) -> None:
        self.w: np.ndarray = None
        self.b: float = 0

    def train(self,
              X: np.ndarray,
              Y: np.ndarray
              ) -> None:
        self.w = np.random.rand(X.shape[1])
        for x, y in zip(X, Y):
            error = y - self.predict(x)
            if error != 0:
                self.w = self.w + error * x
                self.b += error

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        return np.heaviside(self._h(X), 0).astype(int)

    def _h(self,
           X: np.ndarray
           ) -> np.ndarray:
        return X @ self.w + self.b
