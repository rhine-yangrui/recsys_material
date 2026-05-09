"""Multi-Head Attention.
PREFERRED: torch (大厂最常考, 要求手写 reshape + transpose)
流程:
  1) 线性投影 Q,K,V: (B,L,d) -> (B,L,d)
  2) 拆头: (B,L,d) -> (B,L,h,d_h) -> transpose -> (B,h,L,d_h)
  3) scaled dot product attention (在最后两维)
  4) 合头: (B,h,L,d_h) -> (B,L,h,d_h) -> reshape (B,L,d)
  5) 输出线性层 Wo
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_h = d_model // num_heads
        # TODO: self.Wq, Wk, Wv, Wo = Linear(d_model, d_model)*4
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # TODO:
        # Q = self.Wq(x).view(B, L, self.h, self.d_h).transpose(1, 2)  # (B,h,L,d_h)
        # K, V 同理
        # scores = Q @ K.transpose(-2,-1) / sqrt(d_h)
        # 如果 mask: scores.masked_fill_(~mask, -inf)
        # attn = softmax(scores, -1); out = attn @ V
        # out = out.transpose(1,2).contiguous().view(B, L, self.d_model)
        # return self.Wo(out)
        Q = self.Wq(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        K = self.Wk(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        V = self.Wv(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_h)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, L, self.d_model)
        return self.Wo(out)
