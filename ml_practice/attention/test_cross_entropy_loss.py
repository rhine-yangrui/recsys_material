import numpy as np
import torch
import torch.nn.functional as F
from cross_entropy_loss import cross_entropy


def test_matches_torch():
    np.random.seed(0)
    logits = np.random.randn(16, 5).astype(np.float32)
    target = np.random.randint(0, 5, size=16)
    mine = cross_entropy(logits, target)
    ref = F.cross_entropy(torch.tensor(logits), torch.tensor(target)).item()
    assert abs(mine - ref) < 1e-5


if __name__ == "__main__":
    test_matches_torch(); print("CE OK")
