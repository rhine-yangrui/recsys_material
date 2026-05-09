import numpy as np
from linear_regression_sgd import LinearRegressionSGD


def test_fit_linear():
    np.random.seed(0)
    X = np.random.randn(500, 3)
    true_w = np.array([1.5, -2.0, 0.7])
    y = X @ true_w + 0.3 + 0.01 * np.random.randn(500)
    model = LinearRegressionSGD(3, lr=0.05)
    model.fit(X, y, epochs=200, batch_size=32)
    assert np.allclose(model.w, true_w, atol=0.1)
    assert abs(model.b - 0.3) < 0.1


if __name__ == "__main__":
    test_fit_linear(); print("LR SGD OK")
