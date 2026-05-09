# 生成式推荐 (Generative Recommendation) 论文综述与学习笔记

> 面向搜广推 (Search / Ads / Recommendation) 新人的体系化入门 + 进阶资料。
> 目标：读完即可对当前生成式推荐范式有完整认知，并具备冲击大厂实习的知识储备。

---

## 0. 如何使用 / 扩展本文档

本文档被设计为"活文档" (Living Doc)，结构上分三层：

1. **核心论文综述** (第 2~6 章)：按发表时间顺序排列，每篇论文采用统一的 8 小节模板，方便对比阅读。
2. **横向对比与脉络** (第 7~8 章)：把所有论文放在一个时间线和能力矩阵里看。
3. **可扩展板块** (第 9 章及以后)：个人反思、补充论文、术语表、面试问答……新内容请追加到对应板块下，不要打乱核心论文章节的编号。

新增内容时请遵循：

- **新论文** → 在第 6 章后追加 `2.x 论文名` 节，并在第 7 章时间线、第 8 章对比表里同步登记。
- **读后反思 / 笔记** → 写入第 9 章「个人反思与笔记」，按日期或论文分小节。
- **术语 / 缩写** → 加入第 11 章「术语表」。
- **面试题** → 加入第 12 章「面试问答」。
- **相关链接 / 资源** → 加入第 13 章「参考链接」。

每篇论文统一模板：

1. 元信息（作者 / 机构 / 时间 / 链接）
2. 背景与动机
3. 模型架构
4. 实现细节（数据、超参、训练 trick、基础设施）
5. 实验结果（关键指标）
6. 主要贡献
7. 对此前范式的改进
8. 局限性与开放问题
9. 其他要点 / 八卦

---

## 1. 生成式推荐总览（写给完全新人）

### 1.1 传统推荐范式回顾

工业级搜广推系统过去十年的主流范式是 **级联式 (cascade)**：

- **召回 (Retrieval)**：从亿级物料库中粗筛出几千个候选，常用双塔 (DSSM)、i2i、向量检索 (Faiss)。
- **粗排 (Pre-Rank)**：进一步压缩到几百，模型轻量。
- **精排 (Ranking)**：CTR/CVR 多任务模型，代表作 Wide&Deep、DeepFM、DCN、DIN/DIEN、MMoE、PLE。
- **重排 (Re-Rank)**：考虑多样性、上下文、业务规则。

这种 DLRM (Deep Learning Recommendation Model) 范式的痛点：

1. **目标不一致**：四个阶段各自优化，召回不知道精排在做什么。
2. **难以 scale**：参数量主要堆在 Embedding 表，MLP 部分增加参数对指标的边际收益小，**不存在像 NLP 那样清晰的 Scaling Law**。
3. **特征工程重**：依赖大量手工交叉特征。
4. **冷启动差**：纯 ID Embedding 对新物料不友好。

### 1.2 生成式推荐的核心思想

把 NLP 里 "GPT 把 token 一个一个生成出来" 的范式搬到推荐：

- 把 **物品 (item)** 表示成一个或一串 **token**（语义 ID、Semantic ID）。
- 把用户历史行为序列当成 prompt，让一个 **Transformer / LLM** 通过 next-item / next-token prediction **生成** 下一个要推的 item。
- 训练目标统一为序列建模（Cross-Entropy / InfoNCE），天然继承 LLM 的 **Scaling Law** 与 **预训练—微调—对齐** 三段式方法论。

由此衍生出 4 条主要技术路线，本综述涵盖的 5 篇论文恰好覆盖了这 4 条路线的代表作：

| 路线 | 代表 | 一句话 |
|---|---|---|
| Semantic ID + Seq2Seq 生成式检索 | **TIGER** (2023) | 用 RQ-VAE 把 item 编码成 3 个码本 ID，T5 直接生成 |
| 把推荐重写成"序列转导"以获得 Scaling Law | **HSTU** (2024) | Meta 自研注意力，证明推荐也存在 Scaling Law |
| 用预训练 LLM 做层级化推荐 | **HLLM** (2024) | Item-LLM + User-LLM 两段式 |
| 端到端单模型替代级联系统 | **OneRec / OneRec-v2** (2025) | 快手把召回-粗排-精排-重排折成一个 Encoder-Decoder，再演化为 Lazy Decoder-Only |

---

## 2. TIGER — Recommender Systems with Generative Retrieval

### 2.1 元信息
- **作者 / 机构**：Shashank Rajput 等，Google DeepMind
- **发表**：NeurIPS 2023，arXiv 2305.05065 (2023-11)
- **代号含义**：**T**ransformer **I**ndex for **GE**nerative **R**ecommenders

### 2.2 背景与动机
传统召回是 "学一个 query embedding，去 ANN 索引里查最近邻"。问题：

- 索引和模型解耦，**端到端不可微**。
- 物料 ID 用纯 hash / atomic ID，**没有语义**，冷启差。
- 数据稀疏的长尾物品 embedding 训练不充分。

TIGER 提出：**让模型直接"生成"目标物品的 ID**，且这个 ID 是带语义结构的。

### 2.3 模型架构

两阶段：

**阶段一：Semantic ID 生成器（RQ-VAE）**
1. 用 Sentence-T5 把物品文本（标题、品牌、类别、描述）编成 768 维向量。
2. 一个 MLP encoder 压到 32 维 latent z。
3. **Residual Quantization (RQ-VAE)**：3 个码本，每个码本 256 个 codeword。
   - 第 1 层用 z 找最近 codeword c₁，残差 r₁ = z − c₁；
   - 第 2 层量化 r₁ → c₂，残差 r₂ = r₁ − c₂；
   - 第 3 层量化 r₂ → c₃。
   - 物品最终被表示成 3 个整数 (c₁, c₂, c₃)，形成 **粗→细** 的语义层级。
4. 损失 = 重构损失 + commitment loss (β=0.25)。
5. **冲突处理**：若两个不同物品落到同一 (c₁,c₂,c₃)，追加第 4 个"区分位"。

为什么不用纯 K-Means 或 LSH？RQ-VAE 端到端可学、能利用预训练文本表征、码本层级天然提供"由粗到细"的可推广性。

**阶段二：Seq2Seq 生成式召回**
- 用户行为序列 [item₁, item₂, ..., itemₙ] 被展开成 [c₁¹c₂¹c₃¹, c₁²c₂²c₃², ...]，即 3n 个 token。
- 一个标准 T5（4 enc + 4 dec 层，6 头，head dim 64，FFN 1024）做 next-item 生成：输出 (c₁,c₂,c₃) 三元组就是预测的下一个物品。
- 推理时 **Beam Search** 拿 Top-K。

### 2.4 实现细节
- 数据集：Amazon Reviews 三个子集（Beauty / Sports&Outdoors / Toys&Games），各 10K-20K item、几十万交互。
- RQ-VAE：Adagrad，lr=0.4，bs=1024，20k epochs。
- T5：~13M 参数，200k steps。
- 处理 **invalid ID**（生成的 (c₁,c₂,c₃) 在码本里不存在）：beam 中按概率丢弃，实测无效率仅 0.1%–1.6%。

### 2.5 实验结果
- Beauty 数据集：相对 SOTA S³-Rec，**Recall@5 +17.3%、NDCG@5 +29%**。
- Sports / Toys 同样大幅领先 SASRec、BERT4Rec、S³-Rec、FDSA。
- **冷启动**：用 Semantic ID 的 KNN 都比纯 ID KNN 好；TIGER 生成式则进一步提升。
- **多样性可控**：beam search 时调温度 / top-k 即可在 Recall vs Diversity 间权衡。

### 2.6 主要贡献
1. 把"召回索引"变成"语言模型词表"，召回从 ANN 检索升级为 **token 序列生成**。
2. 提出 RQ-VAE 风格的 Semantic ID，解决了"模型生成的 token 必须可枚举"的关键问题。
3. 验证生成式范式在小规模学术数据集已能反超强 baseline，并自带冷启动 + 多样性优势。

### 2.7 对此前范式的改进
- 取代"双塔 + ANN"的两段式检索；
- 取代纯 ID embedding，用语义 token；
- 端到端可微 → 直接受序列建模 scaling 红利。

### 2.8 局限性
- 学术规模数据集（万级 item），未验证亿级商品库；
- 词表是固定的，**新物品需要重新跑 RQ-VAE**；
- 只覆盖召回，未涉及排序/多目标；
- 无在线 A/B 实验。

### 2.9 其他要点
- TIGER 是后来 OneRec 等工业系统 Semantic ID 设计的"祖师爷"。
- "生成式检索"路线在文档检索 (DSI, Differentiable Search Index) 那边几乎同期出现，可对照学习。

---

## 3. HSTU — Actions Speak Louder than Words: Generative Recommenders Scale

### 3.1 元信息
- **作者 / 机构**：Jiaqi Zhai 等，Meta MRS (Monetization & Ranking Systems)
- **发表**：ICML 2024，arXiv 2402.17152 (2024-02)
- **代号**：**H**ierarchical **S**equential **T**ransduction **U**nit

### 3.2 背景与动机
Meta 想回答一个核心问题：**为什么 NLP / CV 的 scaling law 在推荐系统失效？**

观察：DLRM 把大部分参数堆在 Embedding 表（千亿稀疏参数），但 dense 部分小、特征异构、目标繁杂，难以像 LLM 一样 "把模型变大就涨点"。

HSTU 的赌注：**把推荐重写成纯序列转导问题** (Generative Recommenders, GR)，这样模型结构、训练、scaling 都可以套用 Transformer 那一套。

### 3.3 模型架构

**1) 把推荐统一成序列转导**：把所有用户事件（impression、click、like、cart、purchase、search query …）按时间排成一条 token 序列，模型在每个时刻预测"下一个 item / 下一个 action"。这同时统一了召回、排序、多任务。

**2) HSTU 单元（核心创新）**：
对输入 X 计算
$$U, V, Q, K = \text{Split}(\phi_1(f_1(X)))$$
$$A(X)V(X) = \phi_2\!\left(QK^\top + \text{rab}^{p,t}\right)V(X)$$
$$Y = f_2\!\left(\text{Norm}(A(X)V(X)) \odot U(X)\right)$$
要点：
- φ = SiLU，**注意力是 pointwise 而非 softmax**：因为推荐序列里 token 的"强度/重要性"信息会被 softmax 归一化抹掉（softmax 会强制概率和为 1，丢掉绝对量级，而 like vs impression 的差异本质上是绝对量级）。
- 引入 **rab^{p,t}** (Relative Attention Bias)：同时编码相对位置和相对时间间隔，对推荐里"行为时效性"至关重要。
- U 门控（SiLU gating），结合 V 输出 → 类 GLU。
- 激活内存只有 14d（vs Transformer 33d），适合长序列。

**3) Stochastic Length (SL)**：训练时随机截断历史长度，推理时用全长 → 一种针对超长序列的正则化与训练加速。

**4) M-FALCON 推理**：把多个候选 item 拼成一个 micro-batch 共享 user prefix 的 KV cache，**1.5×–2.99× QPS 提升、最高 285× 等效算力**。

### 3.4 实现细节
- 学术数据集：MovieLens-1M / 20M、Amazon Books。
- 工业：Meta 内部 100B+ 样本规模。
- 自研 GPU kernel：HSTU 比 FlashAttention2 + Transformer 在 8192 长度上快 5.3–15.2 倍。
- 模型规模一路推到 **1.5T 参数**，是当时最大的推荐模型。

### 3.5 实验结果
- ML-1M HR@10：SASRec 0.2853 → HSTU 0.3097 → HSTU-large 0.3294。
- Amazon Books、ML-20M 全面领先。
- **首次在推荐系统观察到清晰的 Scaling Law**：HR ∝ ln(compute)，三个数量级算力都成立。
- Meta 线上 A/B：**+12.4%** 主指标（公司级别巨大胜利）。

### 3.6 主要贡献
1. 提出 HSTU，pointwise + 时间偏置 + GLU 门控，是为推荐序列重新设计的注意力。
2. 实证推荐系统也存在 Scaling Law，打开 "推荐大模型" 的口子。
3. M-FALCON 解决了"序列推理 QPS 不够"的工业卡点。

### 3.7 对此前范式的改进
- 取代 DLRM 的"特征拼接 + MLP"为统一序列建模；
- 取代 softmax attention 为更适合推荐的 pointwise attention；
- 第一个让推荐模型像 LLM 一样"越大越强"。

### 3.8 局限性
- 自研 kernel + 自研框架，复现门槛极高；
- 仍是 next-item 监督，多目标（CTR/CVR/时长等）只是隐式融入；
- 无显式对齐 (alignment) 阶段；
- 论文以 retrieval/ranking 内嵌的方式描述，结构图对新人不够友好。

### 3.9 其他要点
- HSTU 的"sparsity exploitation"：用户序列里 95% 的位置是 impression（弱信号），训练时跳过/降采样能换 10x 训练加速。
- 这篇是后续国内外所有"推荐大模型"工作的精神 ancestor。

---

## 4. HLLM — Hierarchical Large Language Model for Sequential Recommendation

### 4.1 元信息
- **作者 / 机构**：Chen et al.，ByteDance / 抖音
- **发表**：arXiv 2409.12740 (2024-09)
- **代号**：**H**ierarchical **LLM**

### 4.2 背景与动机
学术界已有不少"用 LLM 做推荐"的尝试，但效果常常 **不如经典 SASRec**。HLLM 提出三个待回答的问题：
1. 预训练 LLM 的权重对推荐究竟有没有用？
2. 是否需要在推荐数据上继续微调？
3. LLM 能不能像 NLP 一样在推荐任务上展现 scaling？

### 4.3 模型架构

**两层 LLM**：

- **Item LLM**：把物品的标题 + 描述 + 属性等文本喂入 LLM，末尾追加一个特殊 [ITEM] token，取它的最后隐状态当作 **item embedding**。本质上是用 LLM 当一个"语义编码器"，但保留生成能力以便继续预训练。
- **User LLM**：输入是 item embedding 序列（不再是 token id，因此把原 LLM 的 word embedding 层丢掉，但保留 transformer body 的预训练权重），预测下一个 item embedding。

**损失**：
- 生成式（召回风格）：InfoNCE，把下一个真实 item embedding 当正样本，batch 内其它当负样本。
- 判别式（排序风格）：早融合 / 晚融合两种方式把 user repr 与候选 item repr 交互，BCE 损失。
- 联合：L = λ·L_gen + L_cls。

### 4.4 实现细节
- 基座：TinyLlama-1.1B (HLLM-1B)、Baichuan2-7B (HLLM-7B)。
- 数据：PixelRec (200K/1M/8M)、Amazon Books (~700K 用户)。
- **3 阶段工业训练**：
  1. Item LLM 预训练 + 缓存 item embedding；
  2. User LLM 在缓存上训练（极快，因为不用反传到 Item LLM）；
  3. 端到端联调（可选，按算力裁剪）。
- 关键 trick：**item embedding cache**，让 7B 模型也能服务亿级日活。

### 4.5 实验结果
- Pixel8M Recall@10：SASRec 5.083 → HSTU 5.120 → **HLLM-1B 6.129 (+22.93% 相对均值)**，HLLM-7B 进一步 **+169.58%**。
- Amazon Books 同样领先。
- 抖音线上 A/B：核心指标 **+0.705%**（亿级 DAU 场景下非常可观）。
- Ablation：Item LLM 与 User LLM **都** 必须微调；预训练权重 **必须** 保留；文本越丰富越好；序列越长越好（1000 token 时 AUC 0.7458）。

### 4.6 主要贡献
1. 首次系统验证：**预训练 LLM 的语义先验对推荐是真有用的**，前提是 Item / User 两侧都要在推荐数据上继续训练。
2. 提出层级化结构：把"item 内文本建模"与"user 行为序列建模"解耦，实现了 LLM 推荐的可工程化。
3. 工业可落地的 item embedding caching 方案。

### 4.7 对此前范式的改进
- 比直接 prompt LLM 做推荐效果好得多（且更快）。
- 比 HSTU 更直接地利用了开源 LLM 生态。
- 比 TIGER 更强调"利用文本语义而非纯 semantic id"。

### 4.8 局限性
- 强依赖高质量文本描述，对短视频、纯图像物料友好度差；
- 在线服务成本仍高（虽有 cache）；
- 没有显式的 RL / 偏好对齐；
- "scaling law" 在论文里只到 7B，远未到 HSTU 的 1.5T。

### 4.9 其他要点
- HLLM 是字节系把"开源 LLM 拿来即用"思想推到极致的代表。
- 与 Meta 自研基座的 HSTU 形成有趣对比：一个 "造轮子优先"，一个 "复用 LLM 优先"。

---

## 5. OneRec — End-to-End Generative Recommender for Industrial-Scale Short Video

### 5.1 元信息
- **作者 / 机构**：Kuaishou (快手) OneRec Team
- **发表**：arXiv 2506.13695 (2025-09)

### 5.2 背景与动机
快手要回答比 Meta 更激进的问题：**能不能用一个端到端模型直接替换召回-粗排-精排-重排整条级联链路？**

动机：
- 级联系统目标不一致、维护成本高；
- 短视频场景天然多模态、强时效；
- 想直接吃 LLM 的 scaling + post-training (RLHF) 红利。

### 5.3 模型架构

**A) 多模态 Tokenizer**
- 用 miniCPM-V-8B 编码每个视频的 (caption + tag + OCR + ASR + 封面 + 5 帧)，得到 1280 维向量。
- 经过 4 层 QFormer + 4 个 query token，对齐到一个紧凑表示。
- 对齐目标：item-to-item 对比损失 L_I2I + caption 生成损失 L_caption_gen（用 LLaMA3 解码）。
- 然后 **RQ-Kmeans**（用 K-Means 替代 RQ-VAE 训练，避免 codebook collapse）→ 3 层 hierarchical Semantic ID。

**B) Encoder：4 路用户表征**
1. **静态特征**：uid、年龄、性别。
2. **短期行为**：L_s = 20 条最近交互，含 vid、aid、tag、时间戳、播放时长、视频时长、label。
3. **正反馈序列**：L_p = 256，最近的点赞/收藏/完播等正向行为。
4. **终身序列**：L_l = 2000，使用两阶段层级 K-Means + QFormer 压缩到 N_q = 128 token，在保持长期兴趣的同时控制开销。

四路 token 拼接 + 位置编码 → L_enc 层 Transformer (full self-attn) → encoder 输出 z_enc。

**C) Decoder**
- 输入 [BOS, s¹, s², ...]，目标是逐位生成下一个视频的 3 个 Semantic ID。
- 每层：CausalSelfAttn → CrossAttn(z_enc) → **MoE FFN**（top-k 路由，loss-free 负载均衡）。
- 训练目标：标准 next-token prediction (NTP)。

**D) 奖励系统（用于 post-training）**
1. **P-Score**：多任务 BCE 塔，覆盖 ctr / lvtr (long-view) / ltr (like) / vtr (view-through)。
2. **Format Reward**：生成的 Semantic ID 必须落在合法 hash 词表中。
3. **Industrial Reward**：业务规则。

**E) 后训练两件套**
- **RSFT (Reject Sampling Fine-Tuning)**：按播放时长过滤掉最差的 50% 生成样本再做 SFT。
- **ECPO (Early Clipped GRPO)**：对 GRPO 的 ratio 提前裁剪以稳训。

### 5.4 实现细节
- 模型规模：0.015B / 0.121B Dense，0.935B / 2.633B MoE（24 专家，激活 2 或 4）。
- MFU：训练 23.7%、推理 28.8%（相比原精排基线分别 5.2× / 2.6× 提升）。
- **OPEX 仅为级联系统的 10.6%**。
- 集群：90 台 × 8 GPU，400 Gbps NVLink/RDMA，ZeRO-1，BF16。

### 5.5 实验结果
- 离线：在召回、排序两个层级都击败级联系统对应模块。
- 在线：**承接快手主站 + 极速版 25% QPS**；App Stay Time +0.54% / +1.24%；LT7 多个分位均上涨。

### 5.6 主要贡献
1. 业内首个**真正端到端、规模化上线**的生成式推荐系统。
2. 将 LLM 的 SFT + RLHF (RSFT + ECPO) 流程系统迁移到推荐。
3. 提出多路径 user encoder + MoE decoder + 多模态 RQ-Kmeans tokenizer 的工业模板。

### 5.7 对此前范式的改进
- 砍掉级联，目标统一 → 训练-服务一致；
- OPEX 大幅下降；
- 给推荐引入"奖励模型 + RL 对齐"完整方法论。

### 5.8 局限性（也正是 v2 的动机）
- Encoder 占 **97.66% 的 FLOPs**（context 长 → 计算极不平衡）；
- 用 reward model 做 RL 容易出现 reward hacking；
- 采样效率低，每个 prompt 要 rollout 多次。

### 5.9 其他要点
- 它实质上回答了"中国版 GPT-for-Rec 长什么样"。
- 论文里关于 MFU、OPEX、A/B、机房组网细节都很值钱，是面试时聊"工业落地"的金矿。

---

## 6. OneRec-v2 — Lazy Decoder-Only + Preference Alignment

### 6.1 元信息
- **作者 / 机构**：Kuaishou OneRec Team
- **发表**：arXiv 2508.20900 (2025-10)

### 6.2 背景与动机
v1 上线后暴露两大痛点：
1. Encoder-Decoder 架构里 encoder 占据 ~97.66% 的 FLOPs，**算力分配极不均衡**，限制 scaling。
2. RL 阶段依赖 reward model，存在 **reward hacking** 与 **采样效率低** 问题。

### 6.3 模型架构

**创新一：Lazy Decoder-Only**
- **Context Processor**：把异构特征处理成 *L_kv* 组共享的 (K, V) 对。
- 每个 decoder 层 *l* 用 `l_kv = floor(l · L_kv / N_layer)` 索引到对应的 K/V 组。
- Decoder block 顺序：**Lazy Cross-Attn**（无 W_k、W_v，直接复用 context 提供的 K/V）→ Causal Self-Attn → FFN/MoE。
- 配合 **GQA** (G_kv 个 KV 头组 < H_q 个 query 头)。

效果（Table 2，1B 等效）：
- 朴素 Decoder-Only：634.83 GFLOPs；
- Encoder-Decoder 1:1：296.36；
- **Lazy Dec-Only：18.89 GFLOPs**；
- 收敛 loss 与朴素几乎打平 (3.27 vs 3.28)。

总体：**FLOPs −94%，训练资源 −90%，可支撑 16× 大的参数量**。

**Scaling 实验**：0.1B – 8B 拟合 L̂(N) = 3.13 + 3660 / N^0.489；4B MoE (0.5B 激活) 达到 loss 3.22，优于 2B Dense (3.23)。

**训练数据**：使用 2025-08-10 至 2025-08-14 的流式 impression 数据，并采用 **"New Impression Only"** 组织：每个样本只对最新一次曝光的 item 计算 loss，避免重复计算 + 防止时间穿越导致的标签泄漏。

**创新二：偏好对齐 (Preference Alignment)**

抛弃 reward model，直接用 **真实用户反馈**：

- **Duration-Aware Reward Shaping**：
  - 用 F(d) = floor(log_β(d + ε)) 把视频时长分桶（短视频/长视频对完播率敏感度差异大）；
  - 在每个时长桶 b 内，对该用户的所有播放时间 P_{u,b} 排序，计算分位数 q_i = |{p_j ≤ p_i}| / |P_{u,b}|；
  - 优势 A_i = +1 (q_i > τ_B 且非负反馈)，−1 (dislike)，否则 0。

- **GBPO (Gradient-Bounded Policy Optimization)**：
  $$J_{\text{GBPO}}(\theta) = -\mathbb{E}\!\left[\frac{1}{G}\sum_i \frac{\pi_\theta(o_i\mid u)}{\pi'_{\theta_{\text{old}}}(o_i\mid u)} A_i\right]$$
  其中 π′ 用 stop-gradient 构造动态下界：
  - A ≥ 0：π′ = max(π_old, sg(π_θ))；
  - A < 0：π′ = max(π_old, 1 − sg(π_θ))。

  好处：**不裁剪 ratio**（不像 PPO/GRPO 会扔样本），但仍能用 stop-grad 把梯度量级 bound 住，**保留所有样本** → 采样效率拉满。

### 6.4 实现细节
- Dense 与 MoE 双线，最大 4B (0.5B 激活)；
- 训练数据全是真实曝光流，奖励信号完全来自用户行为；
- 训练目标：序列 NTP + GBPO 后训练。

### 6.5 实验结果
- 离线：lazy dec-only 在同算力下 loss 持平甚至更好；
- **在线 A/B（相对 v1）**：快手主站 +0.467%、极速版 +0.741% App Stay Time；
- 多目标无 seesaw（点赞、关注、完播同涨）。

### 6.6 主要贡献
1. **Lazy Decoder-Only**：把"重 encoder"的 FLOPs 折叠进 decoder 的 cross-attn key/value 缓存，实现工业级超高 MFU。
2. **GBPO**：一种适合推荐的、无裁剪、保留全样本的策略优化算法。
3. 用 **真实用户反馈** 替代 reward model，规避 reward hacking。
4. 给出推荐领域 **8B 内的 Scaling Law 拟合**：L̂(N) = E + A / N^α。

### 6.7 对此前范式的改进
- 相对 v1 的 Encoder-Decoder：算力分配合理 16×、参数可继续放大；
- 相对 PPO/GRPO：无 ratio clip，样本利用率高；
- 相对 reward model RL：避免 hacking、对齐更直接。

### 6.8 局限性
- 训练数据组织（New Impression Only）需要强大的流式特征系统，门槛高；
- 对没有海量真实反馈的中小公司不易复制；
- Context Processor 的 KV 分组超参 (L_kv, S_kv, G_kv) 需要细调。

### 6.9 其他要点
- v2 是目前（2025）最值得对照阅读的"推荐大模型 + RL 对齐"工业论文。
- "Lazy" 这个词点题：稀疏的、按需访问的 cross-attention，是把 NLP encoder-decoder 思想"懒化"到推荐场景。

---

## 7. 时间线 (Timeline)

| 时间 | 论文 | 机构 | 关键词 |
|---|---|---|---|
| 2023-11 | TIGER | Google | RQ-VAE, Semantic ID, T5 生成式召回 |
| 2024-02 | HSTU | Meta | Pointwise attention, 推荐 Scaling Law, M-FALCON |
| 2024-09 | HLLM | ByteDance | 预训练 LLM, Item-LLM + User-LLM |
| 2025-09 | OneRec | Kuaishou | 端到端替代级联，多模态 + RSFT/ECPO |
| 2025-10 | OneRec-v2 | Kuaishou | Lazy Decoder-Only + GBPO |

> 新论文请按时间插入，并在第 8 章对比表追加一行。

---

## 8. 横向对比表

| 维度 | TIGER | HSTU | HLLM | OneRec | OneRec-v2 |
|---|---|---|---|---|---|
| 主要任务 | 召回 | 召回+排序统一 | 召回+排序 | 端到端全链路 | 端到端全链路 |
| 物品表示 | RQ-VAE Semantic ID | 原 ID + 时间偏置 | LLM 文本 embedding | 多模态 RQ-Kmeans Semantic ID | 同左 |
| 主干网络 | T5 (~13M) | 自研 HSTU (1.5T 上限) | TinyLlama / Baichuan2 | Enc-Dec + MoE Dec | Lazy Decoder-Only + MoE |
| 训练目标 | NTP | NTP (pointwise attn) | InfoNCE + BCE | NTP + RSFT/ECPO | NTP + GBPO |
| 是否依赖文本 | 强 | 弱 | 强 | 中（多模态） | 中（多模态） |
| Scaling 验证 | 无 | ★★★（10^3 算力跨度） | ★（≤7B） | ★★（≤2.6B） | ★★★（拟合公式 + 8B） |
| 在线 A/B | 无 | Meta +12.4% | 抖音 +0.705% | 快手 25% QPS | v1 基础上 +0.467%/+0.741% |
| 主要不足 | 学术规模 | 自研栈门槛 | 服务成本高 | Encoder FLOPs 失衡 | 数据组织复杂 |

---

## 9. 个人反思与笔记 (TODO)

> 在这里写下你读完每篇论文后的想法、不理解的点、与工作经验的对照。建议每条加日期。

- **2026-04-07**：（示例）TIGER 的 RQ-VAE 与 OneRec 的 RQ-Kmeans 看似只是量化器换了，但 K-Means 避开 codebook collapse 的工程价值非常大，值得自己跑一遍验证。
- ……

---

## 10. 补充阅读 / 相关论文 (TODO)

按主题分组追加：

- **序列推荐前置**：SASRec、BERT4Rec、S³-Rec、FDSA。
- **生成式检索**：DSI (Differentiable Search Index)、NCI、GENRE。
- **多模态推荐**：M6-Rec、VIP5、MMGCN。
- **推荐 RL/对齐**：RLMRec、DPO-Rec。
- **LLM-as-Recommender**：P5、TALLRec、LLaRA、RecAgent。

---

## 11. 术语表 (Glossary)

| 缩写 | 全称 | 一句话 |
|---|---|---|
| DLRM | Deep Learning Recommendation Model | 传统大规模稀疏推荐模型范式 |
| CTR/CVR | Click-Through Rate / Conversion Rate | 点击率 / 转化率 |
| MFU | Model FLOPs Utilization | 算力利用率，越高越省机器 |
| OPEX | Operating Expenditure | 运营成本（机器电费等） |
| LT7 | Lifetime 7 day | 7 日留存等长期指标 |
| RQ-VAE | Residual Quantized VAE | 用残差量化做向量到离散 ID 的编码 |
| Semantic ID | — | 带语义结构的离散物品 ID |
| Beam Search | — | 序列生成时的宽度搜索 |
| InfoNCE | — | 对比学习中的常用损失 |
| GQA | Grouped Query Attention | 多个 query 头共享一组 KV 头 |
| MoE | Mixture of Experts | 多专家路由的稀疏 FFN |
| RLHF | Reinforcement Learning from Human Feedback | 用人类偏好做 RL 对齐 |
| GRPO / GBPO | Group Relative / Gradient-Bounded Policy Optimization | RL 对齐中的策略梯度方法 |
| NTP | Next Token Prediction | 标准自回归目标 |
| ANN | Approximate Nearest Neighbor | 向量检索 |
| QFormer | Querying Transformer | BLIP-2 提出的多模态压缩模块 |

---

## 12. 面试问答 (Interview Q&A，TODO)

按"基础 → 进阶 → 系统设计"三档分层。先列几道引子，自己边读边补。

**基础**

1. 介绍一下传统推荐的级联架构，为什么生成式推荐想干掉它？
2. 什么是 Semantic ID？为什么不用 hash ID？RQ-VAE 与 RQ-Kmeans 的差别？
3. 为什么 HSTU 用 pointwise attention 而不用 softmax？

**进阶**

4. 推荐系统的 Scaling Law 与 NLP 有什么不同？HSTU 与 OneRec-v2 各自给出了什么结论？
5. HLLM 为什么必须微调 Item LLM 与 User LLM 两边？
6. OneRec 的 4 路 encoder 各负责什么？为什么终身行为序列要用 QFormer 压缩？
7. ECPO 与 GBPO 的区别？为什么 GBPO 不需要 ratio clipping？

**系统设计**

8. 如果让你在公司内部从零搭一个生成式召回，你会怎么做物料 tokenization、模型训练、线上服务、A/B 实验、回滚机制？
9. 当用户行为序列长度从 200 增长到 2000，哪些组件最先成为瓶颈？怎么优化？
10. 如果业务 KPI 是 7 日留存而不是当下点击，怎么设计奖励信号？

---

## 13. 参考链接 (TODO)

- TIGER: https://arxiv.org/abs/2305.05065
- HSTU: https://arxiv.org/abs/2402.17152
- HLLM: https://arxiv.org/abs/2409.12740
- OneRec: https://arxiv.org/abs/2506.13695
- OneRec-v2: https://arxiv.org/abs/2508.20900

---

*最后更新：2026-04-07 — 初版，覆盖 5 篇核心论文。*
