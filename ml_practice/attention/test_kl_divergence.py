import numpy as np
from kl_divergence import kl_divergence


def test_same_distribution_is_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert abs(kl_divergence(p, p)) < 1e-8


def test_nonneg():
    p = np.array([0.1, 0.9]); q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) > 0


if __name__ == "__main__":
    test_same_distribution_is_zero(); test_nonneg(); print("KL OK")
