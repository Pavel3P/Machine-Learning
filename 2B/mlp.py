import numpy as np
from misc.activation_functions import \
    sigmoid, sigmoid_derivative


class MLP:
    activation_functions: dict[str: tuple[callable, callable]] = {
        "sigmoid": (np.vectorize(sigmoid), np.vectorize(sigmoid_derivative)),
    }

    def __init__(self,
                 size: list[int],
                 activation: str = "sigmoid",
                 l_rate: float = 0,
                 ) -> None:
        """

        :param size: number of neurons in each layer
                          including input layer
        :param activation:
        :param l_rate: learning rate
        """
        self.size: list[int] = size
        self.weights: list[np.ndarray] = []
        self.l_rate: float = l_rate
        self.activation, self.act_der = self.activation_functions[activation]

    def _forward(self,
                 X: np.ndarray
                 ) -> list[np.ndarray]:
        outputs: list[np.ndarray] = [X]
        for weight_matrix in self.weights:
            output: np.ndarray = self.activation(weight_matrix @ outputs[-1])
            outputs.append(output)

        return outputs

    def _back(self,
              outputs: list[np.ndarray],
              real_y: np.ndarray
              ) -> list[np.ndarray]:
        errors: np.ndarray = outputs[-1] - real_y
        new_weights: list[np.ndarray] = []

        for i in range(len(self.weights))[::-1]:
            new_weight_matrix = self.weights[i] - self.l_rate * np.tensordot(errors, outputs[i-1])
            new_weights.insert(0, new_weight_matrix)
            errors = self.weights[i].T @ errors

        return new_weights[::-1]

    def _random_init(self
                     ) -> list[np.ndarray]:
        weights: list[np.ndarray] = []
        for rows, cols in zip(self.size, self.size[1:]):
            weight_matrix = np.random.uniform(-.5, .5, (rows, cols))
            weights.append(weight_matrix)

        return weights

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              epochs: int = 1
              ) -> None:
        self.weights: list[np.ndarray] = self._random_init()

        for epoch in range(epochs):
            for x, y in zip(X, Y):
                outputs = self._forward(x)
                self.weights = self._back(outputs, y)

    def predict(self,
                X: np.ndarray
                ) -> np.ndarray:
        return self._forward(X)[-1]
