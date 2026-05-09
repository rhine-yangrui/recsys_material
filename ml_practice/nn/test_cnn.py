import torch
from cnn import SimpleCNN


def test_shape():
    m = SimpleCNN(10)
    x = torch.randn(4, 1, 28, 28)
    out = m(x)
    assert out.shape == (4, 10)


if __name__ == "__main__":
    test_shape(); print("CNN OK")
