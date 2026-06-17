# 04 — Attention Mechanism Deep Dive

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/04_attention_mechanism/04_attention_mechanism.ipynb)

> **Time:** ~50 minutes | **Difficulty:** Intermediate–Advanced | **GPU Required:** No

---

## What You'll Learn

Attention is the core operation behind every modern multimodal model. This notebook derives it completely:

- Self-attention from Q, K, V projections
- Scaled dot-product attention and why $\sqrt{d_k}$ scaling is necessary
- Multi-head attention with complexity analysis
- Cross-attention for multimodal fusion
- Build and visualize attention from scratch

---

## Scaled Dot-Product Attention — Full Derivation

### Step 1: Input Projections

Given input sequence $\mathbf{Z} \in \mathbb{R}^{n \times d_{\text{model}}}$, project into queries, keys, and values:

$$
\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}_V
$$

where $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

### Step 2: Attention Scores

For each query position $i$ and key position $j$, the raw score is the dot product:

$$
s_{ij} = \mathbf{q}_i^\top \mathbf{k}_j = \sum_{\ell=1}^{d_k} q_{i\ell} \, k_{j\ell}
$$

In matrix form: $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times n}$.

### Step 3: Scaling by $\sqrt{d_k}$

Assume $q_\ell, k_\ell$ are i.i.d. with $\mathbb{E}[q_\ell] = \mathbb{E}[k_\ell] = 0$ and $\text{Var}(q_\ell) = \text{Var}(k_\ell) = 1$. Then:

$$
\text{Var}(\mathbf{q}_i^\top \mathbf{k}_j) = \sum_{\ell=1}^{d_k} \text{Var}(q_{i\ell} k_{j\ell}) = d_k
$$

Dividing by $\sqrt{d_k}$ normalizes variance to 1, preventing softmax saturation:

$$
\tilde{s}_{ij} = \frac{s_{ij}}{\sqrt{d_k}}
$$

### Step 4: Softmax and Weighted Sum

$$
\alpha_{ij} = \frac{\exp(\tilde{s}_{ij})}{\sum_{j'=1}^{n} \exp(\tilde{s}_{ij'})}, \qquad
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \boldsymbol{\alpha} \mathbf{V}
$$

Each output row is a convex combination of value vectors: $\mathbf{o}_i = \sum_j \alpha_{ij} \mathbf{v}_j$.

---

## Numerical Worked Example ($n=3$, $d_k=2$)

**Given:**

$$
\mathbf{Q} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad
\mathbf{K} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \\ 1 & 0 \end{pmatrix}, \quad
\mathbf{V} = \begin{pmatrix} 2 \\ 0 \\ 1 \end{pmatrix}
$$

**Step 1:** $\mathbf{Q}\mathbf{K}^\top$:

$$
\mathbf{Q}\mathbf{K}^\top = \begin{pmatrix} 1 & 0 & 1 \\ 1 & 0 & 0 \\ 2 & 0 & 1 \end{pmatrix}
$$

**Step 2:** Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

$$
\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{2}} = \begin{pmatrix} 0.707 & 0 & 0.707 \\ 0.707 & 0 & 0 \\ 1.414 & 0 & 0.707 \end{pmatrix}
$$

**Step 3:** Softmax row 0:

$$
\alpha_{0,:} = \text{softmax}([0.707, 0, 0.707]) = \left[\frac{e^{0.707}}{2e^{0.707}}, \frac{1}{2e^{0.707}}, \frac{e^{0.707}}{2e^{0.707}}\right] \approx [0.422, 0.156, 0.422]
$$

**Step 4:** Output token 0:

$$
o_0 = 0.422 \times 2 + 0.156 \times 0 + 0.422 \times 1 = 0.844 + 0.422 = 1.266
$$

---

## Multi-Head Attention

### Derivation

Split $d_{\text{model}}$ into $h$ heads, each with $d_k = d_v = d_{\text{model}}/h$:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^i, \mathbf{K}\mathbf{W}_K^i, \mathbf{V}\mathbf{W}_V^i)
$$

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O
$$

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| $\mathbf{Q}\mathbf{K}^\top$ | $O(n^2 d_k)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| $\boldsymbol{\alpha}\mathbf{V}$ | $O(n^2 d_v)$ | $O(n d_v)$ |
| **Total (single head)** | $O(n^2 d_k)$ | $O(n^2)$ |
| **Total ($h$ heads)** | $O(n^2 d_{\text{model}})$ | $O(n^2 h)$ |

For $n = 197$ (ViT patches + CLS), $d_{\text{model}} = 768$: attention matrix has $197^2 \approx 39{,}000$ entries per head.

**Numerical example:** $n=64$, $d_{\text{model}}=128$, $h=4$, $d_k=32$:

$$
\text{FLOPs} \approx 2 n^2 d_k h = 2 \times 64^2 \times 32 \times 4 = 1{,}048{,}576 \text{ ops per layer}
$$

---

## Cross-Attention (Multimodal)

Text queries image:

$$
\mathbf{Q} = \mathbf{Z}_{\text{txt}} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Z}_{\text{img}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{Z}_{\text{img}} \mathbf{W}_V
$$

Attention matrix $\boldsymbol{\alpha} \in \mathbb{R}^{T \times N}$ where $T$ = text tokens, $N$ = image patches.

**Numerical example:** $T=2$, $N=3$, $d_k=2$:

$$
\mathbf{Q} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
\mathbf{K} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}
$$

$$
\mathbf{Q}\mathbf{K}^\top / \sqrt{2} = \begin{pmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \end{pmatrix}
$$

Row 0 (token "color"): $\alpha_{0,:} \approx [0.333, 0.167, 0.500]$ — highest weight on patch 2.

---

## What You'll Build

- `ScaledDotProductAttention` module from scratch
- Multi-head wrapper with $h=4$ heads
- Attention heatmap visualization
- Cross-attention demo (text → image patches)

---

## Next Step

**[05_transformer_architecture/05_transformer_architecture.ipynb](../05_transformer_architecture/05_transformer_architecture.ipynb)**
