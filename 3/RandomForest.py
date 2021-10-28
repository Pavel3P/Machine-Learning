import numpy as np
from DecisionTree import DecisionTree


class RandomForest:
    def __init__(self,
                 trees_num: int
                 ) -> None:
        self.trees_num: int = trees_num
        self._trees: list[DecisionTree] = []

    def train(self,
              X: np.ndarray,
              Y: np.ndarray
              ) -> None:
        assert X.shape[1] >= self.trees_num

        features: np.ndarray = np.random.choice(X.shape[1], (self.trees_num, X.shape[1] // self.trees_num), False)
        for f in features:
            dt: DecisionTree = DecisionTree(max_depth=len(f))
            dt.train(X[:, f], Y)
            self._trees.append(dt)

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        votes: np.ndarray = np.array([dt.predict(X) for dt in self._trees])

        return np.apply_along_axis(lambda arr: np.bincount(arr).argmax(), 0, votes)
