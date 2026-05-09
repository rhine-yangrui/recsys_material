import numpy as np
from softmax import softmax


def test_basic():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    y = softmax(x, axis=-1)
    assert np.allclose(y.sum(axis=-1), 1.0)
    assert np.allclose(y[1], [1/3, 1/3, 1/3])


def test_overflow():
    x = np.array([1000.0, 1001.0, 1002.0])
    y = softmax(x, axis=-1)
    assert not np.isnan(y).any()
    assert np.isclose(y.sum(), 1.0)


if __name__ == "__main__":
    test_basic(); test_overflow(); print("softmax OK")
