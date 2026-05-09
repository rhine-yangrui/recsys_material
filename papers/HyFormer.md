## 论文背景
背景和OneTrans类似，都是想要突破之前“序列建模+特征交叉“的局限性 。之前框架的局限性主要体现在三个方面：
1. Sequence transformers rely on overly simplified query representations during sequence compression. 这些query tokens一般来自于一些global features或者candidate-related features，非常有限。仅仅增加query的数量会导致线上推理的减速。
2. 压缩后的序列tokens和非序列tokens进行后融合，interactions非常浅，不能有效捕捉特征之间的dependencies。
3. 模型scaling的能力没有被充分利用，只对interaction modules有用，无法提升joint representations的有效性。

## 模型架构
![[HyFormer.pdf#page=4&rect=59,494,556,710&color=yellow|HyFormer, p.4]]
![[HyFormer.pdf#page=4&rect=61,500,319,709&color=yellow|HyFormer, p.4]]
![[HyFormer.pdf#page=4&rect=340,501,553,705&color=yellow|HyFormer, p.4]]