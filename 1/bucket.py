import numpy as np
from regression import LinearRegression

class Bucket:
    def __init__(self, models: list[LinearRegression], weights: np.ndarray) -> None:
        self.models = models
        self.weights = weights
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models]).T

        return predictions @ self.weights
    
    def MSE(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = np.asarray(X)
        return np.square(
            self.predict(X) - Y
        ).mean()
    
    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        ss_res = np.sum((self.predict(X) - Y) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)

        return 1 - ss_res / ss_tot
