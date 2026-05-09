import torch
from cross_attention import CrossAttention


def test_shape():
    ca = CrossAttention(16)
    x = torch.randn(2, 5, 16)
    ctx = torch.randn(2, 9, 16)
    out = ca(x, ctx)
    if isinstance(out, tuple): out = out[0]
    assert out.shape == (2, 5, 16)


if __name__ == "__main__":
    test_shape(); print("cross_attention OK")
