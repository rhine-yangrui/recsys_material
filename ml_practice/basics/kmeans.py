"""K-Means clustering.
PREFERRED: numpy

流程:
  1) 随机选 k 个样本作为初始质心
  2) 重复:
     a) 每个点分配到最近质心 (用 L2 距离)
     b) 更新质心为簇内均值
     c) 如果质心不再变化就停止
"""
import numpy as np


def kmeans(X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4, seed: int = 0):
    """
    return: labels (N,), centers (k, D)
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    # TODO:
    # idx = rng.choice(N, k, replace=False)
    # centers = X[idx].copy()
    # for _ in range(max_iter):
    #     d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (N, k)
    #     labels = d2.argmin(axis=1)
    #     new_centers = np.stack([X[labels == j].mean(0) if (labels==j).any() else centers[j]
    #                             for j in range(k)])
    #     if np.linalg.norm(new_centers - centers) < tol:
    #         centers = new_centers; break
    #     centers = new_centers
    # return labels, centers
    raise NotImplementedError
