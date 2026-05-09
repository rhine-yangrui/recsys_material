"""Numerically stable softmax.
PREFERRED: numpy (面试常让你纯手写, 不用框架)
核心: 先减 max 避免 exp 溢出, 再归一化.
"""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # TODO: 实现数值稳定 softmax
    # 1) x_max = x.max(axis, keepdims=True)
    # 2) e = exp(x - x_max)
    # 3) return e / e.sum(axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


# ---- torch 版本 (可选练习) ----
def softmax_torch(x, dim=-1):
    import torch
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)
