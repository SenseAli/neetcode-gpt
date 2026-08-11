import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z)
        exp_z = np.exp(shifted)
        result = exp_z / np.sum(exp_z)
        return np.round(result, 4)
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass
