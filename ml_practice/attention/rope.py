"""Rotary Position Embedding (RoPE).
PREFERRED: torch (LLaMA 系列标配, 面试常考)

思想: 对 Q, K 的相邻两维 (x_{2i}, x_{2i+1}) 看作复数, 乘上旋转矩阵
  [cos(mθ_i) -sin(mθ_i)]
  [sin(mθ_i)  cos(mθ_i)]
其中 θ_i = 10000^(-2i/d), m 是 token 位置.

好处: q·k 内积只与相对位置 (m-n) 有关 -> 相对位置编码.
"""
import torch


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    """返回 cos, sin, shape (seq_len, head_dim//2)."""
    # TODO:
    # inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # t = torch.arange(seq_len, device=device).float()
    # freqs = torch.outer(t, inv_freq)        # (seq_len, head_dim/2)
    # return freqs.cos(), freqs.sin()
    raise NotImplementedError


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (..., seq_len, head_dim)
    cos/sin: (seq_len, head_dim/2)
    """
    # TODO:
    # x1 = x[..., 0::2]; x2 = x[..., 1::2]
    # x_rot_even = x1 * cos - x2 * sin
    # x_rot_odd  = x1 * sin + x2 * cos
    # out = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
    # return out
    raise NotImplementedError
