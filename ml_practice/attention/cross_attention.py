"""Cross Attention.
PREFERRED: torch
区别: Q 来自 decoder 输入 x, K/V 来自 encoder 输出 context.
"""
import math
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO: Wq, Wk, Wv = Linear(d_model, d_model)*3
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, context, mask=None):
        """
        x:       (B, Lq, d_model)  -> Q
        context: (B, Lk, d_model)  -> K, V
        """
        # TODO: Q=Wq(x); K=Wk(context); V=Wv(context)
        Q = self.Wq(x)
        K, V = self.Wk(context), self.Wv(context)
        d_k = K.shape[-1]
        scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)  # (B, Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = self.Wo(attn @ V)
        return out
