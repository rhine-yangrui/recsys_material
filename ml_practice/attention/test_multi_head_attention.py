import torch
from multi_head_attention import MultiHeadAttention


def test_shape():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 7, 32)
    out = mha(x)
    assert out.shape == (2, 7, 32)


def test_causal_mask():
    mha = MultiHeadAttention(16, 2)
    x = torch.randn(1, 5, 16)
    L = 5
    mask = torch.tril(torch.ones(L, L)).bool()[None, None]  # (1,1,L,L)
    out = mha(x, mask=mask)
    assert out.shape == (1, 5, 16)


if __name__ == "__main__":
    test_shape(); test_causal_mask(); print("multi_head_attention OK")
