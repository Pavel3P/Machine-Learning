import numpy as np


class Perceptron:
    def __init__(
            self,
    ) -> None:
        self.w: np.ndarray = None

    def train(
            self,
            X: np.ndarray,
            Y: np.ndarray
    ) -> None:
        pass

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        pass
