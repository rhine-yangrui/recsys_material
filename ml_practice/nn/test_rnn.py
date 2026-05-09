import numpy as np
from rnn import RNNCell


def test_shape():
    cell = RNNCell(3, 5, 2)
    xs = np.random.randn(7, 3)
    ys, hs = cell.forward(xs)
    assert ys.shape == (7, 2)
    assert hs.shape == (7, 5)


if __name__ == "__main__":
    test_shape(); print("RNN OK")
