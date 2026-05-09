"""Cross Entropy Loss (from scratch).
PREFERRED: numpy (面试要求你手写 log-softmax + NLL, 并说清数值稳定)

输入:
  logits: (N, C)   未归一化
  target: (N,)     类别索引 int
做法:
  log_probs = logits - logsumexp(logits)   <- 数值稳定
  loss = -mean(log_probs[range(N), target])
"""
import numpy as np


def cross_entropy(logits: np.ndarray, target: np.ndarray) -> float:
    # TODO:
    # m = logits.max(axis=1, keepdims=True)
    # log_sum_exp = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    # log_probs = logits - log_sum_exp
    # nll = -log_probs[np.arange(len(target)), target]
    # return nll.mean()
    m = logits.max(axis=1, keepdims=True)
    log_sum_exp = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    log_probs = logits - log_sum_exp
    nll = -log_probs[np.arange(len(target)), target]
    return nll.mean()


# ---- torch 对照 ----
def cross_entropy_torch(logits, target):
    import torch, torch.nn.functional as F
    return F.cross_entropy(logits, target)
