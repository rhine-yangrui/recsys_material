---
tags:
  - 八股
  - 优化器
title: Adam
---

# Adam 优化器（Adaptive Moment Estimation）

- **核心思想**：同时维护梯度的**一阶矩（动量）**和**二阶矩（自适应学习率）**的指数移动平均，是 Momentum + Adagrad 的结合
- **更新规则**：
	- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$（一阶矩，动量项）
	- $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$（二阶矩，自适应学习率项）
	- 偏差修正：$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$，$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
	- $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \cdot \hat{m}_t$
- **典型超参**：$\beta_1=0.9$，$\beta_2=0.999$，$\eta=10^{-3}$
- **优点**：指数衰减移动平均不会单调递减；偏差修正让初期稳定；几乎是深度学习默认优化器

---

## Adagrad vs Adam 一句话区分

[[Adagrad]] "历史全记住"导致学习率只降不升；Adam "近期加权记忆 + 动量"，解决了学习率过早衰减问题。
