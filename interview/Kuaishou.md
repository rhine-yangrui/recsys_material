---
tags:
  - interview
  - recsys
  - kuaishou
company: Kuaishou
position: recommender algorithm
date: 2026.4.9
round: "1"
topics: BERT, rank, TIGER, Huawei Intern, Transformer, FFN, attention, softmax, sqrt(d_k), positional encoding, RQ-VAE encoder-decoder, SID
outcome: Fail
---
JD：
![[快手-推荐算法工程师.png]]
## 面试概况

- **面试官方向**：精排，团队 ~30 人
- **生成式现状**：只在召回层有少量落地，整体仍以传统召粗精链路为主

---

## 技术问题

### 项目：冷启动 / 数据稀疏

**Q：只有 20 个 ground truth，怎么做推荐系统？**

> 

---

### 主题：BERT 架构与训练
[[BERT]]

**Q1：BERT 的整体架构是什么？如何训练（预训练任务）？**

> BERT全名是bidirectional encoder representations from transformers。它是基于transformer架构的，做双向的自注意力机制，把sentences编码成embeddings，，经过预训练之后，在QA等答案提取的任务上和sentiment analysis等分类任务让都表现很好。
> 
> BERT的训练可以分为两个阶段：
> 	第一个阶段是pretrain。用wikipedia和bookcorpus的语料库进行训练。第一个任务是MLM (Masked Language Model）预测句子中某些token，这些token（占15%）被masked成[mask] (80%) / random token (10%) / original token (10%)。MLM是对词表的多分类任务（softmax + 交叉熵），经过softmax生成词表中每个token的概率。第二个任务是NSP（Next Sentence Prediction)，用[CLS]的representation进行二分类的预测，判断是否为上下句。
> 	第二个阶段是fine-tuning，在具体下游任务上用少量标注数据继续训练。

**Q2：BERT 中 `[CLS]`、`[SEP]` 等特殊 token 的作用是什么？句子内部需要加分隔符吗？**

> 输入格式为 [CLS] A [SEP] B [SEP]。[CLS]放在句首，其最终隐藏状态用于分类任务的输出。[SEP]放在每个句子的末尾，标记句子边界。区分token属于句A还是句B靠Segment Embeddings，而非[SEP]本身。句子内部不需要加分隔符。

---

### 主题：Transformer 细节
[[Transformer]]

**Q3：Transformer 的整体架构（Encoder / Decoder / 各层组成）？**

> Input经过embedding matrix，与POS Embedding一起进入Encoder，encoder里面有6个相同的layer，每个layer里面有MHA -> Add(Residual Connection) & Norm(LayerNorm) -> FFN -> Add & Norm
> Output经过embedding matrix，与POS Embedding一起进入Decoder，decoder里面有6个相同的layer，每个layer里面有Masked MHA -> Add & Norm -> Cross Attention (encoder输出作为Key和Value，decode中的x作为Query) -> Add & Norm -> FFN -> Add & Norm
> 最后还有一层pre-softmax linear + softmax。其中presoftmax linear用的matrix跟embedding matrix一样，仅仅在scale上面有区别，后者是前者的$\sqrt{d_{model}}$倍，以避免embedding被POS embedding主导。

**Q4：FFN 由几层组成？每层做了什么？**

> FFN有两层组成，Linear + ReLU + Linear。对每一层中每个向量单独计算（Position-wise）。

**Q5：Attention 的计算流程？**

> $scores = \frac{QK^T}{\sqrt{d_k}},\ attn = \text{softmax}(\text{scores}),\ out = attn @ V$

**Q6：为什么要除以 $\sqrt{d_k}$？若 $d_k$ 变大，softmax 输出会怎么变化？**

> 因为$QK_T$的variance为$d_k$，除以之后变为1，使得概率分布更平滑。若$d_k$变大，softmax输出的概率会更加平滑。

---

### 主题：位置编码

**Q7：常见的位置编码方式有哪些？BERT 使用哪种？**
[[BERT]]

> 一种是类似于Transformer的fixed绝对位置编码，用sin/cos来做。另一种是BERT使用的learned weight matrix，也是绝对位置编码。两种方式效果差不多。前者允许序列长度的扩大，不受矩阵大小限制。

**Q8：给定一条用户行为序列，训练时你想对 item 做位置编码，你会怎么设计？**

> 

---

### 主题：生成式推荐（TIGER / RQ-VAE）

**Q9：RQ-VAE 的 Encoder 和 Decoder 具体架构是什么？**

> 

**Q10：T5 如何生成 SID（Semantic Item ID）？生成过程中概率是如何计算的？**

> 

---

## 手撕算法

**题目：最大连续子数组的乘积**（Maximum Product Subarray）

**思路**

> 

**代码**

```python

```

---

## 反问环节

1. 团队规模与方向分布？
2. 传统推荐方向（精排）有哪些推荐的学习路径？
> RankMixer, OneTrans, Longer, Twin, Twinv2, Star

---

## 复盘 / 总结

> 
