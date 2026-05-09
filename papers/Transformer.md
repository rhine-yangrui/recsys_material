> [!PDF|yellow] [[Transformer.pdf#page=2&selection=107,43,109,28&color=yellow|📖]]
> > Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution.
> 
> 第一个完全基于自注意力机制来计算representation的transduction model。

## Model Architecture
![[Transformer.pdf#page=3&rect=104,366,514,734&color=yellow|📖]]
### Encoder
- N = 6 identical layers
- Each layer has two sub-layers: Multi-head attention & FFN
- Each sublayer is followed by a Residual Connection & Layer Norm:
	$$\text{output} = \text{LayerNorm}\!\left(x + \text{Sublayer}(x)\right)$$
	其中 $x$ 为该子层的输入，$\text{Sublayer}(x)$ 视层而定：
	$$\text{Sublayer}(x) = \begin{cases} \text{MHA}(x) & \text{第一子层（多头自注意力）} \\ \text{FFN}(x) & \text{第二子层（前馈网络）} \end{cases}$$
- 所有子层及 Embedding 的输出维度统一为 $d_{\text{model}} = 512$，保证残差连接时维度匹配。
### Decoder
- N = 6 identical layers
- In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs MHA over the output of the encoder stack
- Self-attention sub-layer was modified to prevent positions from attending to subsequent positions (**Causal Attention**)

### Q、K、V 的来源

设输入序列的表示矩阵为 $X \in \mathbb{R}^{n \times d_{\text{model}}}$，Q、K、V 均由 $X$ 线性投影得到：

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ 为可学习参数。

**Self-Attention vs. Cross-Attention**

| 场景                                  | Q 来源             | K、V 来源                       |
| ----------------------------------- | ---------------- | ---------------------------- |
| Self-Attention（Encoder）             | $X_{\text{enc}}$ | $X_{\text{enc}}$（同一序列）       |
| Masked Self-Attention（Decoder 第一子层） | $X_{\text{dec}}$ | $X_{\text{dec}}$（加 mask）     |
| Cross-Attention（Decoder 第二子层）       | $X_{\text{dec}}$ | $X_{\text{enc}}$（Encoder 输出） |

Cross-Attention 中 Q 来自 Decoder、K/V 来自 Encoder，使 Decoder 每个位置都能"查询"源序列的编码信息。

**完整计算链**

$$X \xrightarrow{W^Q,\,W^K,\,W^V} Q, K, V \xrightarrow{W_i^Q,\,W_i^K,\,W_i^V} Q_i, K_i, V_i \xrightarrow{\text{Attention}} \text{head}_i \xrightarrow{\text{Concat} \to W^O} \text{output}$$

> 实现上两步投影（$X \to Q/K/V$ 和 $Q/K/V \to Q_i/K_i/V_i$）常合并为一步，但概念上是分开的。
> 实际上先得到QKV，然后再把QKV按照num_head reshape就好。

---

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Reason behind scaling the dot products by $\frac{1}{\sqrt{d_k}}$:
		
	**假设**：$q, k \in \mathbb{R}^{d_k}$，每个分量 $q_i, k_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$（均值为 0，方差为 1）。
	
	**分析点积的方差**：
	$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$
	
	对每一项 $q_i k_i$：
	$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0$$
	$$\operatorname{Var}[q_i k_i] = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$
	
	由于各项独立，累加后：
	$$\operatorname{Var}[q \cdot k] = \sum_{i=1}^{d_k} \operatorname{Var}[q_i k_i] = d_k \quad \Longrightarrow \quad \operatorname{Std}[q \cdot k] = \sqrt{d_k}$$
	
	**问题所在**：当 $d_k$ 较大时，点积的量级约为 $\sqrt{d_k}$，输入 softmax 的 logit 差异极大。设两个 logit 分别为 $z_1 \gg z_2$，则：
	$$\text{softmax}(z_1) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} \approx 1, \quad \frac{\partial \,\text{softmax}}{\partial z} \approx 0$$
	
	softmax 进入**饱和区（梯度消失）**，注意力权重几乎退化为 one-hot，无法有效训练。
	
	**缩放的效果**：除以 $\sqrt{d_k}$ 后，方差归一化回 1：
	$$\operatorname{Var}\!\left[\frac{q \cdot k}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1$$
	
	使 softmax 的输入保持在梯度良好的范围内，从而稳定训练。
- 加了scaling之后，softmax分布更平滑；$\sqrt{d_k}$越大，$z$越小，softmax分布越平滑

### Multi-Head Attention

> ([[Transformer.pdf#page=5&selection=2,0,3,32&color=yellow|Transformer, p.5]])
> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

**各矩阵的维度**（原文参数：$h=8$，$d_{\text{model}}=512$）：

| 矩阵      | 维度                              | 作用                    |
| ------- | ------------------------------- | --------------------- |
| $W_i^Q$ | $d_{\text{model}} \times d_k$   | 将输入投影为第 $i$ 个头的 Query |
| $W_i^K$ | $d_{\text{model}} \times d_k$   | 将输入投影为第 $i$ 个头的 Key   |
| $W_i^V$ | $d_{\text{model}} \times d_v$   | 将输入投影为第 $i$ 个头的 Value |
| $W^O$   | $h d_v \times d_{\text{model}}$ | 将所有头的输出拼接后投影回原始维度     |

其中 $d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$。

**数据流维度追踪**（设序列长度为 $n$）：

$$\underbrace{Q, K, V}_{\mathbb{R}^{n \times 512}} \xrightarrow{W_i^Q, W_i^K, W_i^V} \underbrace{Q_i, K_i, V_i}_{\mathbb{R}^{n \times 64}} \xrightarrow{\text{Attention}} \underbrace{\text{head}_i}_{\mathbb{R}^{n \times 64}} \xrightarrow{\text{Concat}} \underbrace{}_{\mathbb{R}^{n \times 512}} \xrightarrow{W^O} \underbrace{\text{output}}_{\mathbb{R}^{n \times 512}}$$

**为什么要多头？**
- 单头 Attention 只能在一个表示子空间中计算相关性
- 每个头通过独立的投影矩阵 $W_i^Q, W_i^K, W_i^V$ 学习不同的子空间，从而捕捉**不同类型的依赖关系**（如语法、语义、指代等）
- 各头输出 Concat 后通过 $W^O$ 融合，综合多角度的注意力信息
- 与单头相比，参数量基本不变（$h$ 个小矩阵 ≈ 1 个大矩阵），但表达能力更强

### Position-wise Feed-Forward Networks (FFN)
$$\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$$
linear transformation + ReLU + linear transformation
对sequence中每个位置的向量分别独立地过同一个MLP

> ([[Transformer.pdf#page=5&selection=192,0,194,1&color=yellow|Transformer, p.5]])
> While the linear transformations are the same across different positions, they use different parameters from layer to layer.

每一层中每个向量过同一个transformation（weights相同），但不同层的FFN的weights不共享。

### Embeddings and Softmax
> ([[Transformer.pdf#page=5&selection=214,49,231,1&color=yellow|Transformer, p.5]])
> we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [ 30 ]. In the embedding layers, we multiply those weights by √dmodel.

input tokens和output tokens通过learned embedding matrix转换成embeddings，共享一套参数

**Weight Tying：三处共享同一矩阵**

设词表大小为 $|V|$，三处矩阵形状如下：

| 位置                               | 作用            | 矩阵形状                                                        |
| -------------------------------- | ------------- | ----------------------------------------------------------- |
| ① Input Embedding（Encoder 输入）    | token id → 向量 | $E \in \mathbb{R}^{\|V\| \times d_{\text{model}}}$          |
| ② Output Embedding（Decoder 输入）   | token id → 向量 | $E$（同上）                                                     |
| ③ Pre-softmax Linear（Decoder 输出） | 向量 → logits   | $W = E^\top \in \mathbb{R}^{d_{\text{model}} \times \|V\|}$ |

③ 的本质：decoder 输出向量 $h$ 与每个 token embedding 做点积，得分最高的即为预测 token：
$$\text{logit}_i = h \cdot E[i,:]^\top = \langle h,\; \text{token}_i\text{的embedding} \rangle$$

直觉上，embedding 矩阵既定义了"token 在语义空间中的坐标"（输入侧），也定义了"输出向量应该有多像这个 token"（输出侧），共享一套坐标系语义一致。此外参数量从 $3|V|d_{\text{model}}$ 降为 $|V|d_{\text{model}}$，起到正则化效果。

**为什么 Embedding 要乘以 $\sqrt{d_{\text{model}}}$？**

Transformer 输入是 embedding 与 positional encoding 直接相加，两者需要量级匹配：

- Positional Encoding 每维为 $\sin/\cos$ 值（绝对值 $\leq 1$），向量范数 $\approx \sqrt{d_{\text{model}}}$
- 由于 Weight Tying，$E$ 需兼顾 pre-softmax linear 的初始化（如 Xavier），每个元素量级 $\sim \frac{1}{\sqrt{d_{\text{model}}}}$，导致 embedding 向量范数 $\approx 1 \ll \sqrt{d_{\text{model}}}$

若不缩放，PE 的信号会完全主导 embedding，词义信息被淹没。乘以 $\sqrt{d_{\text{model}}}$ 后 embedding 范数升至 $\sim \sqrt{d_{\text{model}}}$，与 PE 量级对齐。

> 此缩放**只在 embedding lookup 时做**，pre-softmax linear 直接用原始 $E^\top$，两侧独立控制尺度。


### Positional Encoding
跟BERT训练不一样，transformer用的是不可学习的fixed绝对位置编码（sinusoidal），而不是learned positional embeddings，好处是可以extrapolate到更长的sequences，不局限于定长（比如512）的序列，而且不用训练，可以稍微减少一点参数量。

### Why Self-attention
文章中总结了三个好处：
1. Total computational complexity per layer
2. Amount of computation that can be parallelized
3. Path length between long-range dependencies in the network
![[Transformer.pdf#page=6&rect=107,596,502,686&color=yellow|Transformer, p.6]]
在sequence长度非常长的时候，限制self-attention只考虑一个size为r的neighborhood，来解决时间复杂度的问题。

### Training
#### Training data
翻译任务，Sentences were encoded using byte-pair encoding

**Byte Pair Encoding (BPE)**

BPE 是一种**子词分词算法**，从字符级出发，反复合并语料中最高频的相邻符号对，直到词表达到目标大小 $|V|$。

算法步骤：
1. 初始词表 = 所有字符
2. 统计语料中所有相邻符号对的频次
3. 合并最高频的一对，加入词表
4. 重复 2-3，直到词表达到目标大小

示例：
- 语料：`low, lower, newest, widest`
- 初始：`l o w`, `l o w e r`, `n e w e s t`, `w i d e s t`
- 合并高频对 `es` → 合并 `est` → 最终：`low`, `lower`, `new+est`, `wid+est`

| 问题 | 传统词级分词 | BPE |
| --- | --- | --- |
| 未登录词（OOV） | `<UNK>` 丢失信息 | 拆成已知子词，保留语义 |
| 词表膨胀 | 形态丰富语言词表极大 | 固定 $\|V\|$，可控 |
| 序列长度 | 字符级过长 | 子词长度适中，attention 代价合理 |

直觉：高频词保持完整（`the` → 1 token），低频词拆成有意义子词（`unhappiness` → `un` + `happiness`）。

**在 Transformer 中的应用**

- 英德翻译使用约 **37,000** token 的**共享 BPE 词表**（source + target 合并）
	- 英文德文都用的是拉丁字母；如果是中文则不能共享
- 共享词表是实现 Weight Tying（Encoder/Decoder Embedding 与 Pre-softmax Linear 共享 $E$）的前提：source 和 target 在同一语义空间，才能用同一套 embedding 矩阵

#### Regularization：Dropout（§5.4）

$P_{drop} = 0.1$，施加在**两个位置**：

**① 每个子层输出（Sub-layer Output Dropout）**

$$\text{output} = \text{LayerNorm}\!\left(x + \text{Dropout}(\text{Sublayer}(x))\right)$$

dropout 打在子层输出上，在残差相加**之前**。覆盖范围：

| 位置 | 子层 | 数量 |
| --- | --- | --- |
| Encoder × 6 层 | MHA + FFN | 12 处 |
| Decoder × 6 层 | Masked MHA + Cross-Attention + FFN | 18 处 |

作用：防止残差块之间特征的协同适应（co-adaptation）。

**② Embedding + Positional Encoding 加和（Embedding Dropout）**

$$\text{input} = \text{Dropout}\!\left(E[\text{token}] \times \sqrt{d_{\text{model}}} + \text{PE}(\text{pos})\right)$$

在 Encoder 和 Decoder 输入端各加一次，防止模型过度依赖特定位置或 token 的初始信号。