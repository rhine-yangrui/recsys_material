import torch
from mla import MLA


def test_shape_and_cache():
    m = MLA(32, num_heads=4, d_c=8).eval()
    x = torch.randn(2, 5, 32)
    out, c = m(x)
    assert out.shape == (2, 5, 32)
    assert c.shape == (2, 5, 8)

    # 增量解码: cache 应该累积长度
    out2, c2 = m(torch.randn(2, 1, 32), kv_latent_cache=c)
    assert c2.shape == (2, 6, 8)
    assert out2.shape == (2, 1, 32)


if __name__ == "__main__":
    test_shape_and_cache(); print("MLA OK")
