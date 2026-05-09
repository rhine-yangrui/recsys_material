## Architecture Comparison
![[OneTrans.pdf#page=2&rect=35,525,313,718|OneTrans, p.2]]
- **Traditional: (encode-then-interaction pipeline)** 
	- **Sequence modeling:** encodes user multi-behavior sequences into candidate-aware representations using local attention or Transformer encoders (MHA + FFN)
	- **Feature interaction:** learns high-order crosses among non-sequential user features via factorization, explicit cross networks, or attention over feature groups.
	- **Limitations:** 
		- restricts bidirectional information flow, limiting how static/context features shape sequence representations.
		- fragments execution and increases latency.
- **OneTrans: (unified causal Transformer backbone)**
	- **Unified tokenizer:** converts both *sequential* features and *non-sequential* features into a single token sequence.
	- **Stacked OneTrans blocks:** a Transformer variant to process token sequence.
		- mixed parameterization:
			-  All sequential tokens share a single set of Q/K/V and FFN weights
			- Non-sequential token receives token-specific parameters
- **Main Contributions**:
	1. Unified framework
	2. Customization for recommenders
	3. Efficient training and serving
		- cross-request KV Caching that reuses user-side computations across candidates
		- FlashAttention, mixed-precision training, and half-precision inference
	4. Scaling and deployment

## Related Work（可以去读读看论文）
### Sequence Modeling
#### 最早的DIN & DSIN
user local attention，对每个candidate把用户序列compress到固定长度的vectors，限制了long-range dependency modeling。

#### 序列建模改进 SASRec / BERT4Rec / BST
self-attentive，每个position都能attend所有用户历史，bidirectional masking提升了sample efficiency。

#### 随着scaling laws的发展 LONGER
target ultra-long behavioral histories with efficient attention and serving-friendly designs

### Feature Interaction
#### 经典模型 Wide&Deep, FM/DeepFM, DCN/DCNv2
- efficient low-order or bounded-degree interactions
- **问题：** once the model stacks enough cross layers, adding more stops causes model quality plateaus instead of continuing to improve

### Attention-based Approaches
- AutoInt
- HiFormer

### Scaling up Application
- Wukong: stacks FM-style interaction blocks with linear compression
- RankMixer: parallel token mixing and sparse MoE

### Unified Architecture
- InterFormer: summary-based bidirectional cross architecture that enables mutual signal exchange between the two components
	- Still separate modules
- Generative Recommenders (GRs): frames recommendation as sequential transduction and proposes efficient long-context backbones such as HSTU

## Methodology
![[OneTrans.pdf#page=3&rect=51,433,561,719&color=yellow|OneTrans, p.3]]
### Framework
数学表达：

**任务定义：** 给定用户 $u$，排序模型对每个候选 item $i$ 预测得分：

$$\hat{y}_{u,i} = f_i(\text{NS},\ \text{S};\ \Theta) \tag{1}$$

其中 $\text{NS}$ 为非序列特征（用户、item、上下文），$\text{S}$ 为用户历史行为序列，$\Theta$ 为可训练参数。两个核心预测目标：

$$\text{CTR}_{u,i} = P(\text{click}=1 \mid \text{NS},\ \text{S};\ \Theta)$$

$$\text{CVR}_{u,i} = P(\text{conv}=1 \mid \text{click}=1,\ \text{NS},\ \text{S};\ \Theta) \tag{2}$$

**初始 token 序列：** Sequential 特征 → S-tokens，Non-sequential 特征 → NS-tokens，拼接后输入模型：

$$X^{(0)} = \left[\ \text{S-tokens}\ ;\ \text{NS-tokens}\ \right] \in \mathbb{R}^{(L_S + L_{NS}) \times d} \tag{3}$$

$L_S$ 为 S-token 数量（含 SEP token），$L_{NS}$ 为 NS-token 数量，$d$ 为统一隐向量维度。

**每个 OneTrans Block 的更新（Pre-Norm 残差结构）：**

$$Z^{(n)} = \text{MixedMHA}\!\left(\text{Norm}\!\left(X^{(n-1)}\right)\right) + X^{(n-1)} \tag{4}$$

$$X^{(n)} = \text{MixedFFN}\!\left(\text{Norm}\!\left(Z^{(n)}\right)\right) + Z^{(n)} \tag{5}$$

其中 Norm 为 RMSNorm；MixedMHA 和 MixedFFN 采用 **mixed parameterization**：S-tokens 共享同一套参数，每个 NS-token 有各自独立的参数。

> ([[OneTrans.pdf#page=4&annotation=640R|OneTrans, p.4]])
> This unified design naturally enables (i) intra-sequence interactions within each behavior sequence, (ii) cross-sequence interactions across multiple sequences, (iii) multi-source feature interactions among item, user, and contextual features, and (iv) sequence-feature interactions, all within a single Transformer stack.

对应序列内交互，跨序列交互（比如购买序列和点击序列），多元特征交互，序列与特征交互

### Features & Tokenization
#### Non-Sequential Tokenization
*NS*一般包括numerical inputs (价格，历史统计CTR等等)，categorical inputs (user ID, item category等等)。处理是bucketized / one-hot encoded之后进行编码。因为工业级系统一般由上百条features，我们用两种方法来控制Non-sequential features的数量。
- **Group-wise Tokenizer:** 人工划分语义组 $\{g_1, \cdots, g_{L_{NS}}\}$，每组过一个独立 MLP：

$$\text{NS-tokens} = \left[\text{MLP}_1(\text{concat}(g_1)),\ \cdots,\ \text{MLP}_{L_{NS}}(\text{concat}(g_{L_{NS}}))\right] \tag{6}$$

- **Auto-Split Tokenizer:** 所有特征拼接后过一个 MLP，再均匀切分：

$$\text{NS-tokens} = \text{split}\!\left(\text{MLP}(\text{concat}(\text{NS})),\ L_{NS}\right) \tag{7}$$

Auto-Split 相比 Group-wise 只需一次 dense (全连接) projection，减少了矩阵乘法的次数，也减少了GPU kernel launch 开销。GPU每次kernel launch都有固定开销，合并成一次大矩阵乘法比多次小矩阵乘法更高效，这就是dense projection在工程上的优势。

最终生成$L_{NS}$个tokens，每个token纬度为d
#### Sequential Tokenization
用户有多条行为序列 $\text{S} = \{S_1, \cdots, S_n\}$，每条序列由事件 embedding (item ID + side information like item category and price) 构成：

$$S_i = \{e_{i1}, \cdots, e_{iL_i}\} \tag{8}$$

各序列的 raw embedding 维度可能不同，用序列专属 MLP 投影到统一维度 $d$：

$$\tilde{S}_i = \left[\text{MLP}_i(e_{i1}),\ \cdots,\ \text{MLP}_i(e_{iL_i})\right] \in \mathbb{R}^{L_i \times d} \tag{9}$$

对齐后按 timestamp-aware（按时间交织）或 timestamp-agnostic（按意图强度拼接，purchase → add-to-cart → click）合并，并在序列间插入可学习的 [SEP] token：

$$\text{S-tokens} = \text{Merge}(\tilde{S}_1, \cdots, \tilde{S}_n) \in \mathbb{R}^{L_S \times d}, \quad L_S = \sum_{i=1}^{n} L_i + L_{\text{SEP}} \tag{10}$$
如果是timestamp-aware的话，不用加[SEP]，但对于每个event要加入sequence-type indicator。
如果是timestamp-agnostic，强意图放在前面，让信息从可靠流向不可靠，高质量的购买信号校准低质量的点击信号。

### OneTrans Block
#### Mixed Causal Attention
[[Transformer#Decoder]]
**核心理解：**
- **Head 的切分**：在 $d$ 维度上切，每个 head 操作 $d_h = d/H$ 维子空间，对整条序列（S + NS 所有 token）同时生效
- **Q/K/V 的区分**：在序列位置上区分，S-tokens 用共享矩阵，NS-tokens 用各自独立矩阵
- 两个维度**正交**，互不干扰
- NS-token 能 attend 所有 S-token，是 causal mask 下三角结构自然给出的，无任何额外操作

**公式：**

$$
(q_i,\ k_i,\ v_i) = (W^Q_i x_i,\ W^K_i x_i,\ W^V_i x_i) \tag{11}
$$

$$
W^{\Psi}_i = \begin{cases} W^{\Psi}_S & i \leq L_S \quad \text{（S-tokens 共享）} \\ W^{\Psi}_{NS,i} & i > L_S \quad \text{（每个 NS-token 独立）} \end{cases} \tag{12}
$$

**Causal Mask 结构**（$L_S=4,\ L_{NS}=2$）：

```
           s₁    s₂    s₃    s₄  │  ns₁   ns₂
  s₁   [   ✓     ✗     ✗     ✗   │   ✗     ✗  ]
  s₂   [   ✓     ✓     ✗     ✗   │   ✗     ✗  ]
  s₃   [   ✓     ✓     ✓     ✗   │   ✗     ✗  ]
  s₄   [   ✓     ✓     ✓     ✓   │   ✗     ✗  ]
  ──────────────────────────────────────────────
  ns₁  [   ✓     ✓     ✓     ✓   │   ✓     ✗  ]  ← 能看全部 S
  ns₂  [   ✓     ✓     ✓     ✓   │   ✓     ✓  ]  ← 能看全部 S + ns₁
```

**完整伪代码：**

```python
# 超参数
L_S, L_NS = 1190, 12
L  = L_S + L_NS   # 总 token 数
d  = 64           # token 维度
H  = 4            # head 数
d_h = d // H      # 每个 head 的维度 = 16

# 输入 X: (L, d)，前 L_S 行是 S-tokens，后 L_NS 行是 NS-tokens

# Step 1：混合投影，算每个 token 的 Q、K、V
# S-tokens：共享矩阵 W_S，shape (d, d)
# NS-tokens：独立矩阵 W_NS，shape (L_NS, d, d)

Q, K, V = zeros(L, d), zeros(L, d), zeros(L, d)

Q[:L_S] = X[:L_S] @ W_Q_S        # S-tokens 共享
K[:L_S] = X[:L_S] @ W_K_S
V[:L_S] = X[:L_S] @ W_V_S

for j in range(L_NS):             # NS-tokens 各自独立
    i = L_S + j
    Q[i] = X[i] @ W_Q_NS[j]
    K[i] = X[i] @ W_K_NS[j]
    V[i] = X[i] @ W_V_NS[j]

# Step 2：切分成 H 个 head（在 d 维度上切）
Q = Q.reshape(L, H, d_h)          # (L, H, d_h)
K = K.reshape(L, H, d_h)
V = V.reshape(L, H, d_h)

# Step 3：Causal Mask（下三角）
mask = tril(ones(L, L))
mask[mask == 0] = -inf

# Step 4：每个 head 内做 Attention（并行）
output = zeros(L, H, d_h)
for h in range(H):
    scores  = Q[:, h, :] @ K[:, h, :].T / sqrt(d_h)  # (L, L)
    scores  = scores + mask
    weights = softmax(scores, dim=-1)                  # (L, L)
    output[:, h, :] = weights @ V[:, h, :]             # (L, d_h)

# Step 5：拼回 d 维
output = output.reshape(L, d)
# output[:L_S]  → 更新后的 S-tokens
# output[L_S:]  → 更新后的 NS-tokens（已 attend 全部 S 历史）
```

### Pyramid Stack
设第 $n$ 层输入 token 序列为 $X = \{x_i\}_{i=1}^{L}$，定义 **tail index set**（最近的 $L'$ 个位置）：

$$\mathcal{Q} = \{L - L' + 1,\ \ldots,\ L\}, \quad L' \leq L$$

只有 $\mathcal{Q}$ 中的位置发出 query，复用公式 (12) 的混合参数化：

$$q_i = W^Q_i\, x_i, \quad i \in \mathcal{Q} \tag{14}$$

Key 和 Value 仍对**全部** $L$ 个 token 正常计算。Attention 结束后只保留 $i \in \mathcal{Q}$ 的输出，序列长度从 $L$ 缩减到 $L'$。

**跨层示意**（OneTrans 实现，$L_S: 1190 \to 12$，线性收缩，每层取整到 32 的倍数）：

```
Layer 1:  Q=(L'₁, d)  K/V=(L,   d)  → 输出 L'₁ 个 token
Layer 2:  Q=(L'₂, d)  K/V=(L'₁, d)  → 输出 L'₂ 个 token
...
Layer N:  Q=(L_NS, d) K/V=(L'_{N-1}, d) → 输出 L_NS 个 token（与 NS-tokens 对齐）
```

**两个好处：**
- **Progressive distillation**：长行为历史逐层压缩到末尾的少量 query，信息向 NS-tokens 汇聚
- **Compute efficiency**：attention cost 从 $O(L^2 d)$ 降至 $O(LL'd)$，FFN 随 $L'$ 线性缩放

### Training and Deployment Optimization
S-token计算过的KV全部缓存，对于一次推荐请求 (request)，一个用户对应几百个候选item。这几百个item便在同一个mini-batch中构成了(user, item, label)数据集，排在mini-batch里相邻的位置。S-token只用计算一次。
- **Stage 1 (S-side, once per request)**: causal masking，存下所有S-tokens的key/value pairs和attention outputs。每个request都执行一次
- **Stage 2 (NS-side, per candidate)**: 对于每个候选物品，计算它的NS-tokens，对之前缓存的S-tokens的K/V进行cross attention计算，再传入token-specific FFN layers。如果用到了SIM的话，SIM子序列被聚合到NS-tokens里面，而不是直接替代S-tokens，保证S-side KV caching能够复用。
```
候选 item₁ 的处理：
  SIM 子序列 [鞋A, 鞋F, 袜G]
       ↓ pooling（mean / attention pooling）
  一个 d 维向量
       ↓
  作为 NS-token 之一喂进模型

原来的 NS-tokens = [用户特征, 候选特征, 上下文特征]
现在的 NS-tokens = [用户特征, 候选特征, 上下文特征, SIM聚合向量]
```
- **KV Caching across Requests**: 同一个user behavior sequence是append-only，对于线上推理的时候，我们仍然可以KV Caching across requests。每次只在之前的cache上增加新的behavior计算出的KV值（得益于每次S-side都用的是同一个QKV的矩阵权重）。这样能够把pre-request sequence computation从$O(L)$降低到$O(\Delta L)$，$\Delta L$是新的行为的数量。

#### Unified LLM Optimizations
FlashAttention-2

### Experimental Setup
#### Dataset
Data are split chronologically, with all features snapshotted at impression time to prevent temporal leakage and ensure online-offline consistency.
