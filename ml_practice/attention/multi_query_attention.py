"""Multi-Query Attention (MQA).
PREFERRED: torch
关键: 多个 Query 头, 但所有头共享 1 组 K/V (节省 KV cache 显存).
  Wq: d_model -> d_model    (h 个头)
  Wk: d_model -> d_h        (1 个头)
  Wv: d_model -> d_h        (1 个头)
广播: K/V 的 head 维 = 1, 与 Q 的 head 维广播相乘.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.h = num_heads
        self.d_h = d_model // num_heads
        self.d_model = d_model
        # TODO:
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, self.d_h)
        self.Wv = nn.Linear(d_model, self.d_h)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # TODO:
        # q = self.Wq(x).view(B, L, self.h, self.d_h).transpose(1, 2)   # (B,h,L,d_h)
        # k = self.Wk(x).view(B, L, 1,      self.d_h).transpose(1, 2)   # (B,1,L,d_h)
        # v = self.Wv(x).view(B, L, 1,      self.d_h).transpose(1, 2)
        # scores = q @ k.transpose(-2,-1) / sqrt(d_h)     # 广播到 (B,h,L,L)
        # attn = softmax(scores, -1); out = attn @ v
        # return self.Wo(out.transpose(1,2).reshape(B, L, self.d_model))
        q = self.Wq(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        k = self.Wk(x).view(B, L, 1,      self.d_h).transpose(1, 2) # (B, 1, L, L)
        v = self.Wv(x).view(B, L, 1,      self.d_h).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_h) # 广播到 (B, h, L, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1) # (B, h, L, L)
        out = attn @ v
        return self.Wo(out.transpose(1, 2).contiguous().view(B, L, self.d_model))
