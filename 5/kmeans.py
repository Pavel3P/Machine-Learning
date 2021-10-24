import numpy as np


class KMeans:
    clusters: np.ndarray

    def __init__(self,
                 k: int
                 ) -> None:
        self.k: int = k

    def train(self,
              X: np.ndarray
              ) -> None:
        # Random initialization
        self.clusters = X[np.random.randint(0, len(X), self.k)]
        predictions: np.ndarray = self.predict(X)

        while True:
            for cluster in range(self.k):
                cluster_elements: np.ndarray = X[predictions == cluster]
                self.clusters[cluster, :] = np.mean(cluster_elements, axis=0)

            new_predictions: np.ndarray = self.predict(X)
            if np.all(new_predictions == predictions):
                break
            else:
                predictions = new_predictions

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        return np.array([np.argmin(np.linalg.norm(self.clusters - x, axis=1)) for x in X])
