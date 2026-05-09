import torch
from multi_query_attention import MultiQueryAttention


def test_shape():
    m = MultiQueryAttention(32, 4)
    x = torch.randn(2, 6, 32)
    assert m(x).shape == (2, 6, 32)


if __name__ == "__main__":
    test_shape(); print("MQA OK")
