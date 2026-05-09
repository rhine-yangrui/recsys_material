"""2D Convolution forward (from numpy).
PREFERRED: numpy (这题考的就是手写卷积; torch 一行就能完成无练习价值)

输入: x (N, C_in, H, W), weight (C_out, C_in, kH, kW), bias (C_out,)
输出: (N, C_out, H_out, W_out)  其中 H_out = (H + 2p - kH) / s + 1
经典实现: 双层 for (便于理解), 或 im2col + matmul (高性能).
"""
import numpy as np


def conv2d(x, weight, bias=None, stride: int = 1, padding: int = 0):
    # TODO:
    # 1) 按 padding 给 x 补 0
    # 2) 算 H_out, W_out
    # 3) 三重/四重循环: for n, for co, for i, for j:
    #      out[n,co,i,j] = sum( x[n,:, i*s:i*s+kH, j*s:j*s+kW] * weight[co] ) + b[co]
    raise NotImplementedError
