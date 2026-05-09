import torch
from self_attention import scaled_dot_product_attention, SelfAttention


def test_shape():
    B, L, d = 2, 5, 8
    Q = torch.randn(B, L, d); K = torch.randn(B, L, d); V = torch.randn(B, L, d)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, L, d)
    assert torch.allclose(attn.sum(-1), torch.ones(B, L), atol=1e-5)


def test_causal_mask():
    B, L, d = 1, 4, 4
    Q = K = V = torch.randn(B, L, d)
    mask = torch.tril(torch.ones(L, L)).bool()
    out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
    upper = attn[0].triu(1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_self_attention_module():
    sa = SelfAttention(16)
    x = torch.randn(2, 7, 16)
    assert sa(x)[0].shape == (2, 7, 16)


if __name__ == "__main__":
    test_shape(); test_causal_mask(); test_self_attention_module()
    print("self_attention OK")
