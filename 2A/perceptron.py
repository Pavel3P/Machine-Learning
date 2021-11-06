import numpy as np


class Perceptron:
    """
    Implementation of perceptron with
    Rosenblatt's training algorithm
    for binary classification.
    """

    def __init__(self, l_rate: float = 1) -> None:
        self.l_rate = l_rate

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              min_accuracy: float = .9
              ) -> None:
        # Random initialization
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand()

        while np.mean(self.predict(X) == Y) < min_accuracy:
            for x, y in zip(X, Y):
                error = y - self.predict(x)
                if error != 0:
                    self.w = self.w + error * self.l_rate * x
                    self.b += error

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        return np.heaviside(self._h(X), 0).astype(int)

    def _h(self,
           X: np.ndarray
           ) -> np.ndarray:
        return X @ self.w + self.b
