"""Batch Normalization (training mode).
PREFERRED: numpy 手写 + 说清 train/eval 的差别.

训练:
  mean = x.mean(0); var = x.var(0)
  x_hat = (x - mean) / sqrt(var + eps)
  out   = gamma * x_hat + beta
  running_mean = momentum * running_mean + (1-momentum) * mean
  running_var  = momentum * running_var  + (1-momentum) * var
推理: 使用 running 统计量.
"""
import numpy as np


def batch_norm_train(x, gamma, beta, running_mean, running_var,
                     momentum=0.9, eps=1e-5):
    """
    x: (N, D)
    return: out, running_mean, running_var
    """
    # TODO
    raise NotImplementedError


def batch_norm_eval(x, gamma, beta, running_mean, running_var, eps=1e-5):
    # TODO
    raise NotImplementedError
