import numpy as np
from scipy.optimize import minimize
from node import Node
from typing import Union


class DecisionTree:
    """
    Implementation of Binary Decision Tree
    """

    def __init__(self,
                 max_depth: int
                 ) -> None:
        self._n_classes: int = 0
        self._n_features: int = 0
        self._tree: Node = None
        self.max_depth = max_depth

    def _split_gini(self,
                    Y_left: np.ndarray,
                    Y_right: np.ndarray
                    ) -> float:
        left_gini: float = self._gini(Y_left)
        right_gini: float = self._gini(Y_right)

        return (left_gini * len(Y_left) + right_gini * len(Y_right)) / (len(Y_right) + len(Y_left))

    def _gini(self,
              Y: np.ndarray
              ) -> float:
        if len(Y) == 0:
            return 1

        frequencies: np.ndarray = self._calc_samples_per_class(Y) / len(Y)

        return 1 - np.sum(frequencies ** 2)

    def _split_data(self,
                    X: np.ndarray,
                    Y: np.ndarray,
                    used_features: list[int] = []
                    ) -> Union[tuple[int, float], None]:

        weighted_gini: list[float] = []
        thresholds: list[float] = []
        for feature_index in range(self._n_features):
            minimization = minimize(
                lambda thresh: self._split_gini(Y[X[:, feature_index] <= thresh], Y[X[:, feature_index] > thresh]),
                x0=np.mean(X[:, feature_index])
            )
            weighted_gini.append(minimization.fun)
            thresholds.append(minimization.x)

        for feature in np.argsort(weighted_gini):
            if feature not in used_features:
                return feature, thresholds[feature]

    def _grow_tree(self,
                   X: np.ndarray,
                   Y: np.ndarray,
                   depth: int = 0,
                   used_features: list[int] = []
                   ) -> Node:
        num_samples_per_class: np.ndarray = self._calc_samples_per_class(Y)
        node = Node(
            gini=self._gini(Y),
            num_samples_per_class=num_samples_per_class
        )

        if depth < self.max_depth:
            split = self._split_data(X, Y, used_features)
            if split is not None:
                feature_index, threshold = split
                used_features.append(feature_index)
                node.feature_index = feature_index
                node.threshold = threshold

                left_indices: np.ndarray = X[:, feature_index] <= threshold

                if np.sum(left_indices) > 0:
                    node.left = self._grow_tree(X[left_indices], Y[left_indices], depth+1, used_features.copy())
                if np.sum(~left_indices) > 0:
                    node.right = self._grow_tree(X[~left_indices], Y[~left_indices], depth+1, used_features.copy())

        return node

    def _predict(self,
                 x: np.ndarray
                 ) -> int:
        node: Node = self._tree

        while node is not None:
            prediction: int = node.predicted_class
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return prediction

    def _calc_samples_per_class(self,
                                Y: np.ndarray
                                ) -> np.ndarray:
        return np.array([np.sum(Y == i) for i in range(self._n_classes)])

    def train(self,
              X: np.ndarray,
              Y: np.ndarray
              ) -> None:
        self._n_classes = len(np.unique(Y))
        self._n_features = X.shape[1]

        self._tree = self._grow_tree(X, Y)

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        if len(X.shape) < 2:
            raise ValueError("X should be an array of samples")
        else:
            return np.array([self._predict(x) for x in X])
