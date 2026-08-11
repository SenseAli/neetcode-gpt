import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.array(X)
        weights = np.array(weights)
        predictions = X.dot(weights)
        return np.round(predictions, 5)
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        model_prediction = np.array(model_prediction)
        ground_truth = np.array(ground_truth)
        mse = np.mean((model_prediction - ground_truth) ** 2)
        return round(mse, 5)
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        pass
