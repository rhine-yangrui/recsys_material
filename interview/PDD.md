Abstract中将TIGER定义为一个 **召回层** 的框架
比起之前的序列推荐范式，从ANN search变成了directly predict SID

## Framework
Two stages:
### 1. Semantic ID generation using content features
![[TIGER.pdf#page=4&rect=106,474,506,724&color=yellow|TIGER, p.4]]
- **Item Embeddings**:
	- Each item has associated content features that capture useful semantic information (e.g. titles or descriptions or images). 
	- Content features are embedded by a pre-trained text encoders such as Sentence-T5 or BERT (我的复现中用的nomic-embed-text).
	- Semantic embeddings are quantized to generate an SID for each item.
- **Semantic ID definitions and properties** in this paper
	- a tuple of codewords of length m
	- number of items is equal to the product of the codebook sizes
	- similar items (items with similar semantic embeddings or similar context features) should have overlapping SIDs
- **RQ-VAE** (Residual-Quantized Variational AutoEncoder) for SIDs
	- The auto-encoder is jointly trained by updating the quantization codebook and the DNN encoder-decoder parameters.
	- Why we use a separate codebook of size K for each of the m levels, instead of using a single, mK-sized codebook?
		- Because the norm of residuals tends to decrease with increasing levels, hence allowing for different granularities for different levels.
#### RQ-VAE Framework
- DNN encoder就是最简单的MLP，712 -> 512 -> 256 -> 128 -> 32，中间过程激活函数是ReLU。Decoder大概率是跟encoder对称的结构，解码回768的原始语义嵌入空间。Decoder的唯一目的就是提供训练信号，让我们知道量化后的离散码丢失了多少信息，倒逼Encoder和Codebook学习到高质量的量化表示。
- Residual Quantizer量化层级数 $m = 3$，每层独立codebook，大小 $K = 256$， 每个codebook vector的维度 = 32。每个level的codebook就是一个可学习的矩阵：
	```
	Codebook C_d  ∈  R^{K × d_latent}  =  R^{256 × 32}
	```
- **量化时的查找过程**：给定encoder输出的残差 $r_d \in \mathbb{R}^{32}$，找最近的码本向量：$c_d = \arg\min_k \|r_d - C_d[k]\|^2$。实现上展开距离公式用矩阵运算一次性算完：
	$$\|r_d - C_d[k]\|^2 = \|r_d\|^2 - 2\,r_d^\top C_d[k] + \|C_d[k]\|^2$$
	```python
	# r: [batch, 32],  codebook: [256, 32]
	distances = (
	    (r ** 2).sum(dim=-1, keepdim=True)              # [batch, 1]
	    - 2 * r @ codebook.T                             # [batch, 256]
	    + (codebook ** 2).sum(dim=-1, keepdim=True).T    # [1, 256]
	)
	indices = distances.argmin(dim=-1)  # [batch] ← 这就是 codeword
	```
- **损失函数**：$L(x) = L_{recon} + L_{rqvae}$
	- **Reconstruction loss**（更新 encoder + decoder）：$L_{recon} = \|x - \hat{x}\|^2$
	- **$L_{rqvae}$** 因为量化操作（argmin）不可微，用 stop-gradient (sg) 把联合优化拆成两个方向：
		- **Codebook loss**（只更新 codebook vectors）：$\|\text{sg}[r_i] - e_{c_i}\|^2$，把 encoder 输出当固定目标，拉 codebook 向量靠近残差
		- **Commitment loss**（只更新 encoder）：$\beta\|r_i - \text{sg}[e_{c_i}]\|^2$，把 codebook 向量当固定目标，迫使 encoder 输出稳定在码本向量附近，$\beta = 0.25$
	```
	Encoder 输出 rᵢ  ←── commitment loss ──→  Codebook 向量 e_cᵢ
	  (encoder 往码本靠)                        (码本往 encoder 靠)
	        ↑                                        ↑
	   sg 在 e_cᵢ 上                            sg 在 rᵢ 上
	   只更新 encoder                           只更新 codebook
	```
- **Codebook 的 k-means 初始化**：防止 codebook collapse（随机初始化导致大量码本向量远离数据分布，永远不被选中）。在第一个 training batch 上**逐级串行**初始化：
	1. Level 0：encoder 输出 $z = E(x)$ 上跑 k-means (K=256)，聚类中心作为 $C_0$ 初始值
	2. Level 1：用已初始化的 $C_0$ 对 $z$ 量化，得残差 $r_1 = z - e_{c_0}$，对 $r_1$ 跑 k-means，聚类中心作为 $C_1$ 初始值
	3. Level 2：同理，对 $r_2 = r_1 - e_{c_1}$ 跑 k-means 初始化 $C_2$
	- 必须逐级初始化，因为每级残差的分布和尺度不同（norm 递减）。这保证每个码本向量都落在该级残差的密集区域内，codebook usage 达到 ≥ 80%。
- 码本里面的index是纯粹的整数标签，100、200、201之间没有距离上的意义
- **Handling Collisions**:
	这个是训练后一次性的后处理，训练过程中不关心碰撞
	```
	1. 用训练好的 encoder + quantizer 给所有 item 生成 3-tuple SID
	2. 建一个 lookup table: {SID → [item list]}
	3. 遍历 lookup table:
	   - 如果某个 SID 只对应 1 个 item  → 追加 0 → (c₀, c₁, c₂, 0)
	   - 如果某个 SID 对应 k 个 item   → 分别追加 0,1,...,k-1
	4. 最终每个 item 都有唯一的 4-tuple SID
	```
	注意前3位有语义层级结构，但第4位纯粹是任意分配的序号。
- 新Item的SID编码流程：
	```
	新 item 的 content features
	    │
	    ▼  Sentence-T5（已有，不用重新训练）
	768 维 embedding
	    │
	    ▼  RQ-VAE encoder + quantizer（已训练好，直接推理）
	3-tuple: (c₀, c₁, c₂)
	    │
	    ▼  查 lookup table
	    │
	    ├─ 该 3-tuple 不存在 → 分配 (c₀, c₁, c₂, 0) ✅
	    │
	    └─ 该 3-tuple 已有 k 个 item → 分配 (c₀, c₁, c₂, k) ✅
	```

### 2. Training a generative recommender system on SIDs (§3.2)

#### 核心思路：用 Seq2Seq 直接生成下一个 item 的 Semantic ID
- 传统方法：学 embedding → 建 ANN 索引 → 近邻搜索召回候选
- TIGER：**直接用 Transformer decoder 逐 token 预测下一个 item 的 SID tuple**，把检索变成生成任务

#### 输入序列构造
- 每个用户的交互序列按时间排序：$(\text{item}_1, \dots, \text{item}_n)$，目标是预测 $\text{item}_{n+1}$
- 设 item$_i$ 的 $m$-长 Semantic ID 为 $(c_{i,0}, c_{i,1}, \dots, c_{i,m-1})$，将交互历史展平拼接为：
	$$(\underbrace{c_{1,0}, \dots, c_{1,m-1}}_{\text{item}_1}, \underbrace{c_{2,0}, \dots, c_{2,m-1}}_{\text{item}_2}, \dots, \underbrace{c_{n,0}, \dots, c_{n,m-1}}_{\text{item}_n})$$
- 模型要生成的 target 就是 $\text{item}_{n+1}$ 的 SID：$(c_{n+1,0}, \dots, c_{n+1,m-1})$
- 输入最前面还拼了一个 **user ID token**（用 Hashing Trick 映射到 2000 个桶之一），发现加 user ID 能提升个性化效果

#### Seq2Seq 模型架构（基于 T5X）
- **Encoder-Decoder Transformer**：
	- Encoder 和 Decoder 各 **4 层**
	- **6 个 self-attention heads**，dimension = 64
	- MLP 维度：输入 1024，输出 128
	- 激活函数：ReLU，Dropout = 0.1
	- 总参数量约 **13M**
> ([[TIGER.pdf#page=6&annotation=776R|TIGER, p.6]])
> To allow the model to process the input for the sequential recommendation task, the vocabulary of the sequence-to-sequence model contains the tokens for each semantic codeword. In particular, the vocabulary contains 1024 (256 × 4) tokens to represent items in the corpus. In addition to the semantic codewords for items, we add user-specific tokens to the vocabulary. To keep the vocabulary size limited, we only add 2000 tokens for user IDs.
- **词表设计**：
	- Semantic codeword tokens：$256 \times 4 = 1024$ 个（4 层 codebook，每层 256）
	- User ID tokens：2000 个（通过 Hashing Trick 从原始 user ID 映射）
	- 总词表大小受控，保持轻量

#### 训练细节
- Batch size = 256
- Learning rate = 0.01（前 10k steps），之后用 **inverse square root decay**
- Beauty / Sports and Outdoors 数据集训练 **200k steps**
- Toys and Games 数据集较小，训练 **100k steps**
- 优化器：Adagrad（RQ-VAE 部分用的也是 Adagrad）

#### 生成的 SID 可能不匹配任何 item
- 因为是自回归生成，decoder 输出的 SID tuple 不一定在 item corpus 的 lookup table 中
- 论文 Fig. 6 显示这种"无效生成"的概率很低
- 处理方式见 Appendix E（beam search 时可以用 constrained decoding 等策略）

#### 与传统方法的关键区别
| | 传统 retrieve-and-rank | TIGER generative retrieval |
|---|---|---|
| 表示 | item → 高维连续 embedding | item → 短离散 SID tuple |
| 检索 | ANN (MIPS) | Transformer autoregressive decode |
| 索引 | 显式向量索引（FAISS 等） | Transformer 参数即隐式索引 |
| 新 item | 需重建索引 | 只需生成 SID，无需重建 |
| 存储 | 高维向量，内存开销大 | 整数 tuple + lookup table，极轻量 |

