"""MLP 前向 + 反向 (纯 numpy, 手写 backprop).
PREFERRED: numpy (面试专门考手推反向; torch 版可选看即可)

网络: x -> Linear(d_in, d_hidden) -> ReLU -> Linear(d_hidden, d_out) -> softmax -> CE loss

Backward 关键:
  dL/dz2 = softmax - onehot(y)             (对 logits 的梯度, N 均分)
  dL/dW2 = h^T @ dz2
  dL/dh  = dz2 @ W2^T
  dL/dz1 = dL/dh * (z1 > 0)                (ReLU 导数)
  dL/dW1 = x^T @ dz1
"""
import numpy as np


class MLP:
    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, 0.1, (d_hidden, d_out))
        self.b2 = np.zeros(d_out)

    def forward(self, x):
        # TODO: z1 = x@W1+b1; h = relu(z1); z2 = h@W2+b2
        # 保存中间量用于反向
        raise NotImplementedError

    def backward(self, y_true):
        """y_true: 整数标签 (N,); 返回梯度 dict."""
        # TODO: softmax + CE 的联合梯度
        raise NotImplementedError

    def step(self, grads, lr=0.1):
        # TODO: self.W1 -= lr * grads['W1'] 等
        raise NotImplementedError
