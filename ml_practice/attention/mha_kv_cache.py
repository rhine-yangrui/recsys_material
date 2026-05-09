"""MHA with KV Cache (GPT-2 自回归解码).
PREFERRED: torch (大厂高频, 面试常被问为什么需要 KV cache, 以及 shape)
思路: 推理时每步只传入最新 token (L=1), Q 只算新 token 的, K/V 把历史 cache 拼接上.
    past_k, past_v: (B, h, L_past, d_h)
    new_k, new_v:   (B, h, 1,      d_h)
    k = concat([past_k, new_k], dim=-2)   v 同理
    返回 out 和更新后的 (k, v)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHAWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.h = num_heads
        self.d_h = d_model // num_heads
        self.d_model = d_model
        # TODO: Wq, Wk, Wv, Wo
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        """
        x: (B, L, d_model)  训练时 L>=1, 解码时 L==1
        kv_cache: tuple(past_k, past_v) 或 None
        return: out, (new_k, new_v)
        """
        B, L, _ = x.shape
        # TODO:
        # q = self.Wq(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        # k = self.Wk(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        # v = self.Wv(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        # if kv_cache is not None:
        #     pk, pv = kv_cache
        #     k = torch.cat([pk, k], dim=2)
        #     v = torch.cat([pv, v], dim=2)
        # scores = q @ k.transpose(-2,-1) / sqrt(d_h)
        # attn = softmax(scores, -1); out = attn @ v
        # out = out.transpose(1,2).contiguous().view(B, L, self.d_model)
        # return self.Wo(out), (k, v)
        q = self.Wq(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        k = self.Wk(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        v = self.Wv(x).view(B, L, self.h, self.d_h).transpose(1, 2)
        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        d_k = k.shape[-1]
        scores = q @ k.transpose(-1, -2) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return out, (k, v)
