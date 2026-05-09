---
tags:
  - 八股
  - 量化
  - 生成式推荐
title: RQ-VAE
---

# RQ-VAE（残差量化变分自编码器）

- **核心思想**：先用 Encoder **压缩到低维潜在空间**，再在低维空间做残差量化，Decoder 提供重建监督信号
- **流程**：
	1. Encoder：$x$(768d) → $z$(32d)，降维 + 学习对量化友好的表示
	2. 残差量化：在 32d 空间逐级量化 $z$ → SID = $(c_0, c_1, \dots, c_{m-1})$
	3. Decoder：量化表示 $\hat{z}$ → 重建 $\hat{x}$(768d)
- **损失函数**：$L = L_{recon} + L_{codebook} + L_{commitment}$
	- Reconstruction loss 显式约束量化信息损失
	- Codebook loss + Commitment loss 通过 stop-gradient 联合优化 encoder 与码本
- **优点**：
	- 低维空间聚类更紧凑，量化质量高
	- 端到端联合优化，全局最优
	- 码本向量仅 32d，存储轻量
- **缺点**：训练更复杂，需训练 Encoder/Decoder 网络

---

## 相关

- [[VAE]]：原始 VAE 与 RQ-VAE 的潜在空间形态对比
- [[RQ-KMeans]]
- [[RQ-KMeans vs RQ-VAE]]
- [[papers/TIGER]]：RQ-VAE 在生成式推荐中的应用
