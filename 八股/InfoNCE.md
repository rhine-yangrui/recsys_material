---
tags:
  - 八股
  - 对比学习
  - loss
title: InfoNCE Loss
---

# InfoNCE Loss（Noise Contrastive Estimation）

- **核心思想**：在一批样本中，把一个正样本 $k^+$ 从 $N-1$ 个负样本里"识别出来"，本质是一个 $N$ 分类的交叉熵。通过最大化 query 与正样本的相似度、最小化与负样本的相似度，学到判别性表示
- **公式**：
$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\sum_{i=0}^{N-1} \exp(q \cdot k_i / \tau)}$$
	- $q$：query 向量（anchor）
	- $k^+$：正样本 key；$k_i$：候选集中的第 $i$ 个 key（含 1 个正 + $N-1$ 个负）
	- $\tau$：温度系数（temperature）
- **温度 $\tau$ 的作用**：
	- $\tau$ 小 → softmax 更"尖锐"，对 hard negative 更敏感，梯度更大但训练不稳
	- $\tau$ 大 → 分布更平滑，对负样本区分度下降
	- 经验值：$\tau \in [0.05, 0.2]$（如 SimCLR 0.1，CLIP 可学习初始化 0.07）
- **与互信息的关系**：InfoNCE 是互信息 $I(q; k^+)$ 的一个**下界**，最小化 InfoNCE 等价于最大化 query 与正样本的互信息下界
- **推荐系统中的应用**：
	- **双塔召回**：user tower 输出作 query，item tower 输出作 key，同 batch 内其他 item 当 in-batch negative
	- **对比学习预训练**：SimCSE、CLIP、MoCo 等均使用 InfoNCE
- **工程要点**：
	- **负样本数量**关键：$N$ 越大下界越紧，效果越好 → MoCo 用 queue、CLIP 用大 batch
	- **in-batch negative** 会引入热门 item 过度曝光偏差 → 常加 **logQ correction**（减去采样概率的 log）
	- **L2 归一化** + 温度缩放几乎是标配（让点积等价于余弦相似度）
