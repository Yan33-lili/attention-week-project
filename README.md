# Attention Week Project

## Project Overview

This repository collects the Week 3 Attention Week demos for:

- HuggingFace tokenizer basics
- Transformer tensor-shape flow
- CLIP-VLM image-text similarity

The repository is being prepared as a research-style project. This stage preserves only the demos and outputs that were actually completed.

## Project Structure

```text
attention-week-project/
├── README.md
├── REPORT.md
├── requirements.txt
├── .gitignore
├── configs/
├── src/
├── experiments/
│   └── week3/
├── tests/
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── metrics/
└── docs/
    └── debug_notes.md
```

## Environment

The original project was developed in a Windows + conda workflow. The current code uses `pathlib` and repository-relative paths so the same commands can run on Windows, Linux, or macOS.

- `torch`
- `transformers`
- `pillow`
- `pyyaml`

## Installation

```bash
pip install -r requirements.txt
```

## Available Demos

- `experiments/week3/tokenizer_demo.py`
- `experiments/week3/transformer_shape_demo.py`
- `experiments/week3/clip_vlm_demo.py`

## Basic Usage

Run the demos from the repository root:

```bash
python -m experiments.week3.tokenizer_demo --config configs/tokenizer.yaml
python -m experiments.week3.transformer_shape_demo --config configs/transformer.yaml
python -m experiments.week3.clip_vlm_demo --config configs/clip.yaml
```

The CLIP demo falls back to synthetic local images when `configs/clip.yaml` does not provide `image_inputs`.

## Configs

- `configs/tokenizer.yaml`: tokenizer model name and example texts
- `configs/transformer.yaml`: seed and Transformer shape parameters
- `configs/clip.yaml`: CLIP model name, texts, optional image inputs, and output path

## Outputs

Formal outputs are stored under `outputs/`:

- `outputs/figures/`
- `outputs/logs/`
- `outputs/metrics/`

Current preserved outputs:

- tokenizer output screenshot
- Transformer shape screenshot
- CLIP-VLM success screenshot
- tokenizer log text
- CLIP similarity matrix summary
- attention shape diagram

## Reproducibility Notes

- The completed scope is limited to tokenizer, Transformer shape, CLIP-VLM, and the attention-shape diagram.
- No new experimental results are introduced in this repository stage.
- Some future Attention Week experiments are intentionally not included yet.
- Local validation in the `vla` conda environment succeeded with the three commands above.

## Known Limitations

- Not all planned Attention Week experiments have been implemented.
- The CLIP demo uses a synthetic local-image fallback when image paths are not supplied.
- Full paper-style analysis and broader ablation work are not yet present.


# Multi-Head Self-Attention 手写实现

## 实验简介

本实验不使用 PyTorch 内置的 `MultiheadAttention`、ViT 或 Swin Transformer 模块，而是从零实现 Vision Transformer（ViT）中的核心组件：

**Multi-Head Self-Attention（多头自注意力机制）**

实现内容包括：

- Query、Key、Value 的线性映射
- Scaled Dot-Product Attention（缩放点积注意力）
- Multi-Head Attention 的多头拆分与合并
- Attention Weight（注意力权重）计算
- 输出投影层

整体计算公式如下：

\[
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
\]

其中：

- \(Q\)：Query，表示当前 token 想寻找的信息
- \(K\)：Key，表示 token 可以提供的信息类型
- \(V\)：Value，表示实际传递的信息内容
- \(d_k\)：每个 attention head 中 Query/Key 向量的维度

---

# 文件结构
models/
└── attention.py
|
└── MultiHeadSelfAttention

experiments/
└── week4/
|
└── mha_demo.py


其中：

- `models/attention.py`

  存放手写实现的 Multi-Head Self-Attention 模块。

- `experiments/week4/mha_demo.py`

  用于测试该模块，通过模拟 Vision Transformer 中的 patch embedding 输入验证实现是否正确。

---

# 实验设置

本实验使用模拟的 Vision Transformer patch embedding 作为输入。

参数设置：

- Batch Size：2
- Patch 数量（Token 数量）：64
- Embedding Dimension：256
- Attention Head 数量：8

输入张量形式：
(B, N, D)


其中：

- `B`：batch size，即一次输入的图片数量
- `N`：图片被划分后的 patch/token 数量
- `D`：每个 patch 的特征维度

该形式对应 ViT 中经过 Patch Embedding 后的输入格式。

---

# 运行结果

## 1. 输入（Input）

运行输出：
torch.Size([2, 64, 256])


含义：

- `2`：当前 batch 中包含 2 张图片
- `64`：每张图片被划分为 64 个 patch/token
- `256`：每个 patch 使用 256 维向量表示


该输入表示：
2张图片

↓

每张图片64个视觉token

↓

每个token由256维特征表示


符合 Vision Transformer 中 patch embedding 后的数据结构。

---

## 2. 输出（Output）

运行输出：
torch.Size([2, 64, 256])


输出保持与输入相同的维度。

原因：

Self-Attention 不会改变：

- token 数量
- embedding 维度

它的作用是：

通过计算不同 patch 之间的注意力权重，让每个 patch 融合其他 patch 的上下文信息。

因此：

输入时：
Patch_i = 当前区域的局部视觉信息


经过 Attention 后：
Patch_i = 当前区域信息 + 其他相关patch提供的上下文信息


每个 patch 的表示能力得到增强。

---

## 3. Attention Map（注意力矩阵）

运行输出：
torch.Size([2, 8, 64, 64])



该张量表示 Multi-Head Self-Attention 中不同 patch 之间的信息交互关系。


维度含义：



(batch_size, num_heads, query_tokens, key_tokens)



对应：


- `2`：batch size
- `8`：attention head 数量
- `64`：Query token 数量
- `64`：Key token 数量




因此：


每一个 attention head 都会产生一个：



64 × 64



的注意力矩阵。




矩阵中的元素：


\[
Attention(i,j)
\]


表示：


> 在更新第 i 个 patch 的表示时，第 j 个 patch 对它的重要程度。




例如：


如果：



Attention(1,5)=0.8



表示：


第1个patch在更新自身特征时，会大量参考第5个patch提供的信息。


---


# 实现验证总结


本实验成功从零实现了 Multi-Head Self-Attention，并验证了其核心流程。


实现包括：


## 1. Query、Key、Value生成


通过可学习线性层：


\[
Q=XW_Q
\]


\[
K=XW_K
\]


\[
V=XW_V
\]




将输入 token 映射到不同的信息空间。




---


## 2. Scaled Dot-Product Attention


计算 Query 与 Key 的匹配程度：


\[
Score=\frac{QK^T}{\sqrt{d_k}}
\]




然后通过 Softmax 得到 Attention Weight：


\[
A=softmax(Score)
\]




---


## 3. Value 信息聚合


利用注意力权重融合不同 token 的信息：


\[
Output=AV
\]




使每个 patch 获得来自其他 patch 的上下文信息。




---


## 4. Multi-Head Attention


将 embedding 维度拆分为多个 attention head：



256维embedding

↓

8个head

↓

每个head独立计算attention

↓

拼接(concat)

↓

输出投影



不同 head 可以学习不同类型的视觉关系，例如：


- 局部纹理关系
- 物体结构关系
- 长距离语义关系


---


# 实验结论


通过本实验，成功复现了 Vision Transformer 中最核心的 Self-Attention 模块。


实验输出：



Input:
torch.Size([2,64,256])

Output:
torch.Size([2,64,256])

Attention Map:
torch.Size([2,8,64,64])



符合标准 Transformer Attention 工作流程。


该实现验证了：


- Patch token 可以通过 Attention 进行全局信息交互
- 每个 patch 可以动态选择需要关注的其他 patch
- Multi-Head Attention 能够学习不同空间关系


该模块对应 MiniViT-Full 中 Transformer Encoder 的核心计算部分。