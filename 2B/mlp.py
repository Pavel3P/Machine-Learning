import numpy as np
from misc.activation_functions import \
    sigmoid, sigmoid_derivative


class MLP:
    """
    Implementation of Multilayer Perceptron
    with backpropagation training function
    """

    activation_functions: dict[str: tuple[callable, callable]] = {
        "sigmoid": (np.vectorize(sigmoid), np.vectorize(sigmoid_derivative)),
    }

    def __init__(self,
                 size: list[int],
                 threshold: float = .5,
                 activation: str = "sigmoid",
                 l_rate: float = 1,
                 ) -> None:
        """

        :param size: number of neurons in each layer including input layer
        :param threshold: parameter for thresholding activation function
                          in prediction function
        :param activation: type of activation function
        :param l_rate: learning rate
        """
        self.size: list[int] = size
        self.weights: list[np.ndarray] = []
        self.threshold: float = threshold
        self.l_rate: float = l_rate
        self.activation, self.act_der = self.activation_functions[activation]

    def _forward(self,
                 x: np.ndarray
                 ) -> list[np.ndarray]:
        outputs: list[np.ndarray] = [np.array([x]).T]
        for weight_matrix in self.weights:
            output: np.ndarray = self.activation(weight_matrix @ outputs[-1])
            outputs.append(output)

        return outputs

    def _back(self,
              outputs: list[np.ndarray],
              real_y: np.ndarray
              ) -> list[np.ndarray]:
        errors: np.ndarray = outputs[-1] - real_y.reshape(outputs[-1].shape)
        new_weights: list[np.ndarray] = []

        for i in range(len(self.weights))[::-1]:
            new_weight_matrix = self.weights[i] - self.l_rate * np.tensordot(
                                                                        errors.reshape(errors.shape[0]),
                                                                        outputs[i].reshape(outputs[i].shape[0]),
                                                                        axes=0
                                                                            )
            new_weights.insert(0, new_weight_matrix)
            errors = self.weights[i].T @ errors

        return new_weights

    def _random_init(self
                     ) -> list[np.ndarray]:
        weights: list[np.ndarray] = []
        for cols, rows in zip(self.size, self.size[1:]):
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
            idxs: np.ndarray = np.arange(len(X))
            np.random.shuffle(idxs)

            for x, y in zip(X[idxs], Y[idxs]):
                outputs = self._forward(x)
                self.weights = self._back(outputs, y)

    def predict(self,
                x: np.ndarray
                ) -> np.ndarray:
        return (self._forward(x)[-1] > self.threshold).astype(int)
