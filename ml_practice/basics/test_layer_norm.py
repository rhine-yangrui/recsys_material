import numpy as np
import torch
import torch.nn.functional as F
from layer_norm import layer_norm


def test_matches_torch():
    np.random.seed(0)
    x = np.random.randn(4, 8).astype(np.float32)
    g = np.ones(8, dtype=np.float32); b = np.zeros(8, dtype=np.float32)
    mine = layer_norm(x, g, b)
    ref = F.layer_norm(torch.tensor(x), (8,)).numpy()
    assert np.allclose(mine, ref, atol=1e-5)


if __name__ == "__main__":
    test_matches_torch(); print("LN OK")
