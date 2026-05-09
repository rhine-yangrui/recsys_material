"""Multi-head Latent Attention (MLA, DeepSeek-V2).
PREFERRED: torch

核心思想:
  把 K,V 低秩压缩成一个小维度 latent c_kv (B, L, d_c),
  解码时只缓存 c_kv 而不是完整 K,V, 显存减少数十倍.
  真正算注意力时再用 W_UK, W_UV 把 c_kv 投回 head 空间.

简化版框架 (不含 RoPE 解耦, 便于练习):
  c_kv = x @ W_DKV                 # (B, L, d_c)      <- 这是要 cache 的
  K    = c_kv @ W_UK               # (B, L, h*d_h)
  V    = c_kv @ W_UV               # (B, L, h*d_h)
  Q    = x @ W_Q  (可选再经过一个压缩 c_q -> W_UQ)
  然后 reshape 成多头, 做标准 scaled dot product.

进阶:
  完整 MLA 里 Q/K 还要拆成 "content" 和 "rope" 两部分, rope 部分单独保留一份小 K,
  这样才能和 RoPE 兼容. 这个框架先不要求你写那块, 先把 latent 压缩跑通.
"""
import math
import torch
import torch.nn as nn


class MLA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_c: int):
        super().__init__()
        self.h = num_heads
        self.d_h = d_model // num_heads
        self.d_model = d_model
        self.d_c = d_c
        # TODO:
        # self.W_DKV = nn.Linear(d_model, d_c, bias=False)   # down-projection
        # self.W_UK  = nn.Linear(d_c, d_model, bias=False)   # up K
        # self.W_UV  = nn.Linear(d_c, d_model, bias=False)   # up V
        # self.W_Q   = nn.Linear(d_model, d_model, bias=False)
        # self.W_O   = nn.Linear(d_model, d_model, bias=False)
        raise NotImplementedError

    def forward(self, x, kv_latent_cache=None):
        """
        x: (B, L, d_model)
        kv_latent_cache: (B, L_past, d_c) 或 None — MLA 真正要 cache 的东西就这么小
        return: out, new_c_kv
        """
        B, L, _ = x.shape
        # TODO:
        # c_kv = self.W_DKV(x)                           # (B, L, d_c)
        # if kv_latent_cache is not None:
        #     c_kv = torch.cat([kv_latent_cache, c_kv], dim=1)
        # K = self.W_UK(c_kv).view(B, -1, self.h, self.d_h).transpose(1,2)
        # V = self.W_UV(c_kv).view(B, -1, self.h, self.d_h).transpose(1,2)
        # Q = self.W_Q(x).view(B, L, self.h, self.d_h).transpose(1,2)
        # attn = softmax(Q @ K.transpose(-2,-1) / sqrt(d_h), -1)
        # out = (attn @ V).transpose(1,2).reshape(B, L, self.d_model)
        # return self.W_O(out), c_kv
        raise NotImplementedError
