Stands for **Bidirectional Encoder Representations from Transformers**

> [!PDF|255, 208, 0] [[BERT.pdf#page=1&annotation=865R|BERT, p.1]]
> > Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying finetuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

BERT criticises **fine-tuned token-level tasks**, where the model must label tokens within the input itself and being unable to look right is genuinely harmful. ChatGPT sidesteps this by framing QA as generation rather than span labeling.

## Comparison
**Feature-based** approach keeps the pre-trained model frozen — its output embeddings are extracted and passed as fixed features into a separate task-specific architecture (e.g., a BiLSTM). ELMo is the canonical example.

**Fine-tuning** approach directly adapts the pre-trained model on the downstream task — a minimal task-specific head is added, and *all* weights are updated end-to-end. GPT exemplifies this before BERT.

BERT's critique: both prior approaches relied on **unidirectional** pre-training, which is tolerable for sentence-level tasks but harmful for token-level ones (NER, QA) where a token's label depends on context from both sides. BERT resolves this via **Masked Language Modelling (MLM)**, enabling true bidirectionality, then applies fine-tuning — same architecture, state-of-the-art on both task types.

## 预训练阶段 (Pre-train)

![[BERT.pdf#page=3&rect=69,593,529,786|BERT, p.3]]
- start/end span **不是加到输入里的特殊 token**，而是 QA 任务的**输出机制**：
	输入格式为 `[CLS] 问题 [SEP] 段落 [SEP]`，BERT 对段落中的**每一个 token** 输出两个分数：
	- **Start logit**：这个 token 是答案起始位置的概率
	- **End logit**：这个 token 是答案结束位置的概率
	取分数最高的一对 (i, j)（要求 i ≤ j），对应的 span 就是预测的答案片段。这也印证了 BERT 对单向模型的批评——判断一个 token 是否是答案的起点，必须同时看它左边和右边的内容。

- **`[CLS]`** (CLassification Segment)：每个序列的第一个 token，其最终隐藏状态作为整句的聚合表示，用于分类任务的输出。
- **`[SEP]`**：分隔符 token，用于区分句对（句 A 与句 B 之间，以及序列末尾各插入一个）。配合 Segment Embeddings（`E_A` / `E_B`）告诉模型每个 token 属于哪句话。
- **NSP**（Next Sentence Prediction）：预训练任务之一。输入句对，预测 B 是否是 A 的下一句（50% 真实连续，50% 随机替换），让模型学习句间关系。

> [!PDF|yellow] [[BERT.pdf#page=4&selection=19,0,80,1&color=yellow|BERT, p.4]]
> > We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B. As shown in Figure 1, we denote input embedding as E, the final hidden vector of the special [CLS] token as C ∈ RH , and the final hidden vector for the ith input token as Ti ∈ RH .
> 

![[BERT.pdf#page=5&rect=109,668,479,785&color=yellow|BERT, p.5]]

- **Tokenization（WordPiece）**：BERT 使用 **WordPiece** 分词，词表约 30,000。优先将完整词映射为单个 token；若词不在词表中则递归拆成子词（subword），延续片段加 `##` 前缀（如 `"playing"` → `["play", "##ing"]`）。完整流程：
	1. 原始文本按空格 / 标点做基础切分
	2. 各词经 WordPiece 拆分，映射为词表 ID
	3. 首位插入 `[CLS]`，句段末尾插入 `[SEP]`
	4. 三类 Embedding **逐位叠加**后输入 Encoder：
		- **Token Embedding**：token 本身的语义向量
		- **Segment Embedding**：标记属于句 A（`E_A`）还是句 B（`E_B`）
		- **Position Embedding**：token 在序列中的绝对位置
		三类 Embedding 共用同一套底层机制，本质都是**查表（lookup）**：
		**① Token 里存的是什么**
		WordPiece 分词后，每个 token 被映射为词表中的一个整数 ID（索引），如 `[CLS]` → `101`，`play` → `2377`，`##ing` → `4917`。token 本身不含任何语义，只是词表里的一个编号。

		**② Token → Token Embedding**
		模型维护一个可学习的**嵌入矩阵** $E \in \mathbb{R}^{V \times d}$（$V \approx 30{,}000$，$d = 768$）。将 token ID 作为行索引直接取出对应行向量，即得到该 token 的 768 维语义向量。这一步等价于 `nn.Embedding(vocab_size, hidden_dim)`，在训练中与模型其余参数一同反向传播更新。

		**③ Segment Embedding 的做法**
		同样是一张小的可学习查找表，只有两行：$E_A$（句 A）和 $E_B$（句 B）。输入序列中每个 token 根据其所属句子取对应行向量。对于单句任务，全部 token 一律取 $E_A$。最终三类向量**逐元素相加**（维度相同，均为 768），叠加结果送入第一层 Transformer Encoder：$$\text{input}_i = \text{TokenEmb}(id_i) + \text{SegEmb}(seg_i) + \text{PosEmb}(i)$$

		**④ Position Embedding 的做法**
		BERT 选择**可学习的绝对位置编码（Learned Absolute Positional Embedding）**，而非原版 Transformer 的正弦/余弦固定公式。本质同样是一张查找表：$E_{pos} \in \mathbb{R}^{512 \times d}$，共 512 行（对应最大序列长度），每行是一个 $d=768$ 维向量，随机初始化后在预训练中端到端更新，实现等价于 `nn.Embedding(512, hidden_dim)`。`[CLS]` 占位置 0，后续 token 依次递增。
		- **上限 512**：位置 513 以后没有对应向量，无法处理超长序列（这是 Longformer 等模型的改进动机之一）。
		- **与原版 Transformer 对比**([[Transformer]])：原版用固定公式 $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$，无需参数且理论上可外推更长序列；BERT 的 learned 版本多出 $512 \times d$ 个参数，但实现更简单，论文实验表明两者效果相近。

- **MLM mask 机制**：随机选取输入中 **15%** 的 token 进行遮蔽，让模型预测原词。但若全部替换为 `[MASK]`，微调阶段从未见过该 token 会造成分布偏移，因此对被选中的 15% 采用混合策略：
	- **80%** 替换为 `[MASK]`
	- **10%** 替换为随机 token
	- **10%** 保持原词不变
	模型无法预知哪个位置会被改动，被迫对**每个** token 都维持完整的双向上下文理解。损失函数只计算被 mask 的位置，不计算其余 token。

## 微调阶段 (Fine-tune)

> [!PDF|yellow] [[BERT.pdf#page=5&selection=252,21,261,31&color=yellow|BERT, p.5]]
> > At the output, the token representations are fed into an output layer for tokenlevel tasks, such as sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis.
> 

针对不同的任务，用不同部分的representations：
1. **Sequence Tagging / QA (token-level)**: 把all token representations放进一个output layer
2. **Sentiment Analysis / Entailment**: 把[CLS] representation投入一个output layer for classification