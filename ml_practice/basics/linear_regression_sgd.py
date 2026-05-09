"""Linear Regression with SGD (from numpy).
PREFERRED: numpy (这题本来就考 numpy)

模型: y = X @ w + b
Loss: MSE = mean( (y_hat - y)^2 )
梯度:
  dL/dw = 2/B * X^T (y_hat - y)
  dL/db = 2/B * sum(y_hat - y)
"""
import numpy as np


class LinearRegressionSGD:
    def __init__(self, n_features: int, lr: float = 0.01):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: return X @ self.w + self.b
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        # TODO: 每个 epoch shuffle, 按 batch 计算梯度并更新 self.w, self.b
        raise NotImplementedError
