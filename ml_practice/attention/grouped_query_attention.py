"""Grouped-Query Attention (GQA).
PREFERRED: torch  (LLaMA-2/3 用的就是它, 大厂高频)
关键: num_kv_heads 介于 1 (MQA) 与 num_heads (MHA) 之间.
  每 num_heads/num_kv_heads 个 query 头共享一组 K/V.
实现: k,v 形状 (B, num_kv_heads, L, d_h), 用 repeat_interleave 扩展到 num_heads 再做注意力.
"""
import math
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.h = num_heads
        self.kv_h = num_kv_heads
        self.d_h = d_model // num_heads
        self.d_model = d_model
        self.group = num_heads // num_kv_heads
        # TODO:
        # self.Wq = Linear(d_model, d_model)
        # self.Wk = Linear(d_model, num_kv_heads * self.d_h)
        # self.Wv = Linear(d_model, num_kv_heads * self.d_h)
        # self.Wo = Linear(d_model, d_model)
        raise NotImplementedError

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # TODO:
        # q = Wq(x).view(B, L, h, d_h).transpose(1,2)
        # k = Wk(x).view(B, L, kv_h, d_h).transpose(1,2)
        # v = Wv(x).view(B, L, kv_h, d_h).transpose(1,2)
        # k = k.repeat_interleave(self.group, dim=1)   # -> (B,h,L,d_h)
        # v = v.repeat_interleave(self.group, dim=1)
        # 剩下同 MHA
        raise NotImplementedError
