import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
                              x = np.array(x)
                              W1 = np.array(W1)
                              b1 = np.array(b1)
                              W2 = np.array(W2)
                              b2 = np.array(b2)
                              y_true = np.array(y_true)
                              z1 = W1 @ x + b1
                              a1 = np.maximum(0,z1)
                              z2 = W2 @ a1 + b2
                              y_pred = z2

                              n = len(y_true)
                              loss = np.mean((y_pred - y_true) ** 2)

                              dL_dz2 = (2.0 / n) * (y_pred - y_true)

                              dW2 = np.outer(dL_dz2, a1)
                              
                              db2 = dL_dz2

                              dL_da1 = W2.T @ dL_dz2
                              relu_mask = (z1 > 0).astype(float)
                              dL_dz1 = dL_da1 * relu_mask

                              dW1 = np.outer(dL_dz1, x)

                              db1 = dL_dz1

                              return { 
                                'loss': round(loss, 4),
                                'dW1': np.round(dW1, 4).tolist(),
                                'db1': np.round(db1, 4).tolist(),
                                'dW2': np.round(dW2, 4).tolist(),
                                'db2': np.round(db2, 4).tolist(),
                              }
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
    pass
