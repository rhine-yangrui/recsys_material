# AI 手撕算法练习题

每道题在 `solution` 里有 `TODO`，运行对应的 `test_*.py` 即可验证。
`# PREFERRED: torch/numpy` 注释告诉你大厂面试里更常考哪种写法。

```
python -m pytest test_xxx.py      # 或直接 python test_xxx.py
```

## 注意力篇 (attention/)
1. `softmax.py` — 数值稳定 softmax（减最大值）
2. `self_attention.py` — Scaled Dot-Product + Self Attention
3. `cross_attention.py` — Cross Attention
4. `multi_head_attention.py` — MHA
5. `mha_kv_cache.py` — GPT-2 解码 KV Cache
6. `multi_query_attention.py` — MQA
7. `grouped_query_attention.py` — GQA
8. `mla.py` — DeepSeek Multi-head Latent Attention
9. `rope.py` — 旋转位置编码
10. `cross_entropy_loss.py` — 交叉熵
11. `kl_divergence.py` — KL 散度

## 基础 ML (basics/)
12. `linear_regression_sgd.py` — SGD 线性回归（numpy）
13. `kmeans.py` — K-Means
14. `layer_norm.py`
15. `batch_norm.py`

## 神经网络 (nn/)
16. `mlp.py` — MLP 前向 + 反向（backprop）
17. `conv2d.py` — 二维卷积
18. `cnn.py` — 简易 CNN
19. `rnn.py` — RNN Cell
20. `lstm.py` — LSTM Cell

推荐练习顺序：softmax → self_attention → MHA → KV Cache → MQA/GQA → RoPE → MLA；
基础篇从 linear_regression_sgd 开始；神经网络从 mlp (backprop) → conv2d → rnn → lstm。
