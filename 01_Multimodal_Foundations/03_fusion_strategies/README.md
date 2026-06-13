# 03 — Fusion Strategies: How to Combine Modalities

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/03_fusion_strategies/03_fusion_strategies.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

Five strategies for combining image and text representations — each with different trade-offs:

| Strategy | When Features Merge | Complexity | Interaction Depth | Best For |
|----------|-------------------|-----------|-------------------|----------|
| Early Fusion | Before encoding | Low | Deep (full model) | Simple tasks, small models |
| Late Fusion | After encoding | Low | None | Retrieval, pre-computation |
| Cross-Attention | During encoding | High | Rich, asymmetric | Generation, VQA |
| Gated Fusion | Learned gate | Medium | Adaptive | Noisy/missing modalities |
| Bilinear Fusion | Outer product | High | Fine-grained | Detailed reasoning |

---

## Strategy 1: Early Fusion — Concatenate Then Encode

Combine raw (or lightly processed) features **before** the main encoder:

```
  Image patches          Text tokens
  [p₁, p₂, ..., pₙ]     [t₁, t₂, ..., tₘ]
       │                       │
       └───────┬───────────────┘
               │  Concatenate
               ▼
  [p₁, ..., pₙ, t₁, ..., tₘ]    ← single sequence
               │
               ▼
       ┌──────────────┐
       │  Shared       │
       │  Transformer  │   Full self-attention across both
       │  Encoder      │   modalities from layer 1
       └──────┬───────┘
              │
              ▼
       Joint representation
```

$$
Z_{\text{fused}} = \text{Transformer}([\mathbf{z}_1^{\text{img}}, \ldots, \mathbf{z}_N^{\text{img}}, \mathbf{z}_1^{\text{txt}}, \ldots, \mathbf{z}_M^{\text{txt}}])
$$

**Pros:** Maximum interaction — every image patch can attend to every text token from layer 1.
**Cons:** Quadratic cost $O((N+M)^2)$, no independent pre-computation, modality-specific pre-training is harder.

---

## Strategy 2: Late Fusion — Encode Then Combine

Each modality is encoded **independently**, then combined at the very end:

```
  Image ──► [Image Encoder] ──► v ∈ ℝᵈ ──┐
                                           ├──► combine ──► output
  Text  ──► [Text Encoder]  ──► t ∈ ℝᵈ ──┘
```

**Combination options:**

| Method | Formula | Use Case |
|--------|---------|----------|
| Dot product | $s = \mathbf{v}^\top \mathbf{t}$ | Retrieval ranking |
| Cosine | $s = \frac{\mathbf{v}^\top \mathbf{t}}{\lVert\mathbf{v}\rVert \lVert\mathbf{t}\rVert}$ | CLIP similarity |
| Concatenation | $\mathbf{z} = [\mathbf{v}; \mathbf{t}]$ | Classification |
| Element-wise | $\mathbf{z} = \mathbf{v} \odot \mathbf{t}$ | Simple fusion |
| Addition | $\mathbf{z} = \mathbf{v} + \mathbf{t}$ | Residual fusion |

**Pros:** Each encoder can be pre-trained independently; image/text representations can be pre-computed and cached for fast retrieval.
**Cons:** No cross-modal interaction during encoding — each modality is "blind" to the other.

---

## Strategy 3: Cross-Attention — The Most Powerful Strategy

One modality **queries** the other — "text asks questions, image provides answers":

```
  Text features (queries)     Image features (keys & values)
  Q = Z_txt · W_Q             K = Z_img · W_K
                               V = Z_img · W_V
       │                            │
       └────────┬───────────────────┘
                │
                ▼
  CrossAttn = softmax(Q K⊤ / √d_k) · V

  "Each text token attends to relevant image regions"
```

### Cross-Attention Formula

$$
\text{CrossAttn}(Z_{\text{txt}}, Z_{\text{img}}) = \text{softmax}\!\left(\frac{(Z_{\text{txt}} W_Q)(Z_{\text{img}} W_K)^\top}{\sqrt{d_k}}\right)(Z_{\text{img}} W_V)
$$

**Key insight:** Queries come from one modality, Keys and Values from another. This creates an **asymmetric** interaction — the text "reads" from the image.

### Attention Weight Interpretation

The attention weights $\alpha_{ij}$ form a matrix $\in \mathbb{R}^{M \times N}$ (text tokens $\times$ image patches):

$$
\alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{l=1}^{N} \exp(q_i^\top k_l / \sqrt{d_k})}
$$

Each row sums to 1: for text token $i$, $\alpha_{ij}$ tells us how much it "looks at" image patch $j$.

```
  Attention matrix (text × image patches):

               Image patches (14×14 grid)
            p₁   p₂   p₃  ...  p₁₉₆
  "a"     [ 0.01 0.01 0.02 ... 0.01 ]
  "cat"   [ 0.01 0.35 0.28 ... 0.01 ]  ← attends to cat region
  "on"    [ 0.01 0.01 0.01 ... 0.01 ]
  "mat"   [ 0.01 0.01 0.01 ... 0.42 ]  ← attends to mat region
```

---

## Strategy 4: Gated Fusion — Learned Modality Weighting

A **sigmoid gate** learns how much to trust each modality:

$$
g = \sigma(W_g [\mathbf{v}; \mathbf{t}] + b_g)
$$

$$
\mathbf{z}_{\text{fused}} = g \odot \mathbf{v} + (1 - g) \odot \mathbf{t}
$$

where $g \in (0, 1)^d$ is a per-dimension gate vector, $\sigma$ is the sigmoid function, and $[\mathbf{v}; \mathbf{t}]$ is concatenation.

```
  v ──┐                    ┌──► g ⊙ v ──┐
      ├──► [W_g] ──► σ ──►│              ├──► z_fused
  t ──┘                    └──► (1-g) ⊙ t┘
```

**Why gating?** In real-world data, modalities can be **noisy or missing**. The gate learns to suppress unreliable modalities:
- Blurry image → $g \to 0$ (trust text more)
- Ambiguous text → $g \to 1$ (trust image more)

---

## Strategy 5: Bilinear Fusion — Full Interaction Tensor

Captures **pairwise interactions** between every dimension of both modalities:

$$
z_k = \mathbf{v}^\top W_k \mathbf{t} + b_k, \quad k = 1, \ldots, d_{\text{out}}
$$

Full bilinear tensor form:

$$
\mathbf{z} = \mathbf{v}^\top \mathcal{W} \mathbf{t} \in \mathbb{R}^{d_{\text{out}}}
$$

where $\mathcal{W} \in \mathbb{R}^{d_v \times d_t \times d_{\text{out}}}$.

**Problem:** $d_v \times d_t \times d_{\text{out}}$ parameters — for $d_v = d_t = 768, d_{\text{out}} = 512$: **302M parameters** just for fusion!

### Low-Rank Bilinear (MLB) — Practical Solution

Factor $\mathcal{W}$ with low-rank projections:

$$
\mathbf{z} = (U \mathbf{v}) \odot (V \mathbf{t})
$$

where $U \in \mathbb{R}^{r \times d_v}$, $V \in \mathbb{R}^{r \times d_t}$, and $r \ll d$.

**Parameter reduction:** From $d_v \times d_t \times d_{\text{out}}$ to $r(d_v + d_t)$.
With $r = 64$: $64 \times (768 + 768) = 98{,}304$ — a **3000× reduction**.

---

## Decision Guide — Which Fusion to Use?

```
  Need to pre-compute embeddings for retrieval?
   │
   ├── YES → Late Fusion (CLIP-style)
   │
   └── NO → Need fine-grained cross-modal reasoning?
              │
              ├── YES → Cross-Attention (LLaVA, Flamingo)
              │
              └── NO → Modalities might be noisy/missing?
                         │
                         ├── YES → Gated Fusion
                         │
                         └── NO → Budget / simplicity preference?
                                    │
                                    ├── Simple → Early Fusion
                                    └── Rich   → Bilinear / MLB
```

---

## Computational Cost Comparison

| Strategy | Params Added | Time Complexity | Pre-computable? |
|----------|-------------|----------------|-----------------|
| Early | $O(D^2)$ (shared layers) | $O((N+M)^2 D)$ | No |
| Late | $O(D)$ (projection) | $O(N^2 D + M^2 D)$ | Yes |
| Cross-Attention | $O(D^2)$ (Q/K/V) | $O(NMD)$ | Partially |
| Gated | $O(D^2)$ (gate) | $O(D^2)$ | No |
| Bilinear (full) | $O(D^2 d_{\text{out}})$ | $O(D^2)$ | No |
| MLB (low-rank) | $O(rD)$ | $O(rD)$ | No |

---

## What You'll Build

- All 5 fusion strategies implemented from scratch
- Side-by-side comparison on the same data
- Visualization of cross-attention weights (which image regions attend to which words)
- Performance comparison chart

---

## Prerequisites

- Notebook 02 (ViT and text encoder internals)
- Understanding of softmax and matrix multiplication
- Familiarity with gating mechanisms (sigmoid)

---

## Next Step

**[01_clip_from_scratch](../../02_Vision_Language_Models/01_clip_from_scratch/)** — Build CLIP end-to-end
