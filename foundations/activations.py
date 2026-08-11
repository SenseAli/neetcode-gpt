import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.round(1 / (1 + np.exp(-x)), 5)
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        pass

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        pass