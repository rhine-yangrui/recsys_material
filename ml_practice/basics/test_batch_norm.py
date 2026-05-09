import numpy as np
from batch_norm import batch_norm_train, batch_norm_eval


def test_train_stats():
    np.random.seed(0)
    x = np.random.randn(32, 4).astype(np.float32) * 3 + 1
    g = np.ones(4, dtype=np.float32); b = np.zeros(4, dtype=np.float32)
    rm = np.zeros(4, dtype=np.float32); rv = np.ones(4, dtype=np.float32)
    out, rm2, rv2 = batch_norm_train(x, g, b, rm, rv)
    assert np.allclose(out.mean(0), 0, atol=1e-5)
    assert np.allclose(out.std(0), 1, atol=1e-2)


if __name__ == "__main__":
    test_train_stats(); print("BN OK")
