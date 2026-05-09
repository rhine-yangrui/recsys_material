import numpy as np
from lstm import LSTMCell


def test_shape():
    cell = LSTMCell(4, 6)
    xs = np.random.randn(10, 4)
    hs = cell.forward(xs)
    assert hs.shape == (10, 6)


if __name__ == "__main__":
    test_shape(); print("LSTM OK")
