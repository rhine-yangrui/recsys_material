import torch
from grouped_query_attention import GroupedQueryAttention


def test_shape():
    m = GroupedQueryAttention(32, num_heads=8, num_kv_heads=2)
    x = torch.randn(2, 6, 32)
    assert m(x).shape == (2, 6, 32)


if __name__ == "__main__":
    test_shape(); print("GQA OK")
