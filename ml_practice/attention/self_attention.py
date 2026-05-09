"""Scaled Dot-Product Attention + Self-Attention.
PREFERRED: torch (面试最常考 torch 版; numpy 版留作理解)
公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
Self-Attention: Q,K,V 都来自同一个 x 经过三个线性层.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, ..., Lq, d_k)
    K: (B, ..., Lk, d_k)
    V: (B, ..., Lk, d_v)
    mask: broadcastable to (B, ..., Lq, Lk); True/1 位置保留, False/0 置 -inf
    return: (B, ..., Lq, d_v), attn_weights
    """
    # TODO:
    # 1) scores = Q @ K.transpose(-2,-1) / sqrt(d_k)
    # 2) 如果 mask 存在, 用 masked_fill 把 0 位置填成 -inf
    # 3) attn = softmax(scores, dim=-1)
    # 4) out = attn @ V
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k) # (B, Lq, d_k) @ (B, d_k, Lk) = (B, Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf')) # 位置为0的填充成 -inf
    attn = F.softmax(scores, dim=-1) # (B, Lq, Lk)
    out = attn @ V # (B, Lq, Lk) @ (B, Lk, d_v) = (B, Lq, d_v)
    return out, attn

class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO: 定义 Wq, Wk, Wv 三个 nn.Linear(d_model, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # TODO: Q,K,V = Wq(x), Wk(x), Wv(x); 调用 scaled_dot_product_attention
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        return scaled_dot_product_attention(Q, K, V, mask)
