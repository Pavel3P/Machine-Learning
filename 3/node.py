import numpy as np


class Node:
    def __init__(self,
                 gini: float,
                 num_samples_per_class: np.ndarray,
                 ) -> None:
        self.gini: float = gini
        self.num_samples_per_class: np.ndarray = num_samples_per_class
        self.predicted_class: int = np.argmax(num_samples_per_class)

        self.feature_index: int = 0
        self.threshold: float = 0

        self.left: Node = None
        self.right: Node = None
