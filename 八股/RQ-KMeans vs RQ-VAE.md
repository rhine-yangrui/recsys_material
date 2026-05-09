---
tags:
  - 八股
  - 量化
  - 生成式推荐
title: RQ-KMeans vs RQ-VAE 对比
---

# RQ-KMeans vs RQ-VAE 对比

| | [[RQ-KMeans]] | [[RQ-VAE]] |
|---|---|---|
| 神经网络 | ❌ 无 | ✅ Encoder + Decoder |
| 量化空间 | 原始 embedding 空间（768d） | 学到的潜在空间（32d） |
| 优化方式 | 逐级贪心 K-Means | 端到端联合优化（梯度下降） |
| 信息保留 | 无重建监督，信息损失不可控 | Reconstruction loss 显式约束 |
| 码本向量维度 | 768d（高存储） | 32d（轻量） |
| 高维适应性 | 受维度灾难影响 | Encoder 降维后规避 |
| 实现复杂度 | 简单 | 较复杂 |
| 一句话 | 原始空间硬聚类 | 先学低维表示再聚类，重建损失保证质量 |
