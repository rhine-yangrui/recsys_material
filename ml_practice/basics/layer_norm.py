"""Layer Normalization.
PREFERRED: numpy (面试要求手推, 也要能写 torch 版)

对最后一个维度做 normalization:
  mean = x.mean(-1, keepdims=True)
  var  = x.var(-1, keepdims=True)
  y    = (x - mean) / sqrt(var + eps) * gamma + beta

LN vs BN 区别: LN 在特征维度归一化, 不依赖 batch, 因此适合 NLP/变长序列.
"""
import numpy as np


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # TODO
    raise NotImplementedError
