import numpy as np


def calculate_rmse(prediction: np.array, target: np.array) -> np.array:
    return np.sqrt(np.sum((prediction - target) ** 2, axis=1) / target.shape[1])
