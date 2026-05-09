"""KL Divergence KL(P || Q) = sum P * (log P - log Q).
PREFERRED: numpy (手写), 但要记住 torch 里 F.kl_div(log_q, p) 用的是 log 输入.

两种常见输入:
  1) P, Q 已经是概率分布
  2) 输入是 logits, 需要先 softmax
这里实现 1); 输入概率.
"""
import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12, axis: int = -1) -> np.ndarray:
    # TODO:
    # return np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=axis)
    raise NotImplementedError
