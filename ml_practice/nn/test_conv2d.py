import numpy as np
import torch
import torch.nn.functional as F
from conv2d import conv2d


def test_matches_torch():
    np.random.seed(0)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    mine = conv2d(x, w, b, stride=2, padding=1)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b),
                   stride=2, padding=1).numpy()
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-4)


if __name__ == "__main__":
    test_matches_torch(); print("conv2d OK")
