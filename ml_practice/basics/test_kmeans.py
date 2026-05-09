import numpy as np
from kmeans import kmeans


def test_three_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[0, 0], scale=0.1, size=(50, 2))
    b = rng.normal(loc=[5, 5], scale=0.1, size=(50, 2))
    c = rng.normal(loc=[0, 5], scale=0.1, size=(50, 2))
    X = np.concatenate([a, b, c])
    labels, centers = kmeans(X, k=3, seed=1)
    assert centers.shape == (3, 2)
    # 每个簇里应占绝大多数点同一个 label
    for seg in [slice(0,50), slice(50,100), slice(100,150)]:
        counts = np.bincount(labels[seg], minlength=3)
        assert counts.max() > 40


if __name__ == "__main__":
    test_three_blobs(); print("kmeans OK")
