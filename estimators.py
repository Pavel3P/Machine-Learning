import numpy as np


def accuracy(
        real_y: np.ndarray,
        predicted_y: np.ndarray
) -> np.float64:
    return np.mean(real_y == predicted_y)


def precision(
        real_y: np.ndarray,
        predicted_y: np.ndarray
) -> np.float64:
    true_positive = np.sum((real_y == 1) & (predicted_y == 1))
    false_positive = np.sum((real_y == 0) & (predicted_y == 1))

    return true_positive / (true_positive + false_positive)


def recall(
        real_y: np.ndarray,
        predicted_y: np.ndarray
) -> np.float64:
    true_positive = np.sum((real_y == 1) & (predicted_y == 1))
    false_negative = np.sum((real_y == 1) & (predicted_y == 0))

    return true_positive / (true_positive + false_negative)


def contingency_table(
        real_y: np.ndarray,
        predicted_y: np.ndarray
) -> np.ndarray:
    """
    
    :param real_y: real labels 
    :param predicted_y: predicted labels
    :return: contingency table with structure as given:
             
             True positive  |  False negative
             ---------------|----------------
             False positive |  True negative

    Label 1 is regarded as positive and 0 as negative.
    """

    true_positive = np.sum((real_y == 1) & (predicted_y == 1))
    true_negative = np.sum((real_y == 0) & (predicted_y == 0))
    false_positive = np.sum((real_y == 0) & (predicted_y == 1))
    false_negative = np.sum((real_y == 1) & (predicted_y == 0))

    return np.array([
        [true_positive, false_negative],
        [false_positive, true_negative]
    ])
