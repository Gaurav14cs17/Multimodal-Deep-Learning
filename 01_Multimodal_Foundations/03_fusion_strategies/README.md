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

### Cross-Attention — Step-by-Step Numerical Example

**Setup:** $T=2$ text tokens, $N=3$ image patches, $d_k=2$.

**Step 1:** Project to Q, K, V:

$$
\mathbf{Q} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
\mathbf{K} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad
\mathbf{V} = \begin{pmatrix} 2 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}
$$

**Step 2:** Compute scores $\mathbf{Q}\mathbf{K}^\top / \sqrt{2}$:

$$
\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{2}} = \begin{pmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \end{pmatrix}
$$

**Step 3:** Softmax row 0 (token "color"):

$$
\alpha_{0,:} = \text{softmax}([0.707, 0, 0.707]) \approx [0.422, 0.156, 0.422]
$$

**Step 4:** Weighted values for token 0:

$$
\mathbf{o}_0 = 0.422 \begin{pmatrix} 2 \\ 0 \end{pmatrix} + 0.156 \begin{pmatrix} 0 \\ 1 \end{pmatrix} + 0.422 \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1.266 \\ 0.578 \end{pmatrix}
$$

Token 0 attends equally to patches 0 and 2 (both score 0.707), ignoring patch 1.

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

## Mathematical Proofs

### Proof: Cross-Attention — Q from One Modality, K and V from Another

**Claim:** Cross-attention lets modality A query modality B by using $Q$ from A and $K, V$ from B.

**Step 1 — Define modality-specific projections:**

Given text features $Z_{\text{txt}} \in \mathbb{R}^{M \times D}$ and image features $Z_{\text{img}} \in \mathbb{R}^{N \times D}$:

$$
Q = Z_{\text{txt}} W_Q, \quad K = Z_{\text{img}} W_K, \quad V = Z_{\text{img}} W_V
$$

**Why:** Queries encode "what the text is looking for"; keys/values encode "what the image offers at each patch."

**Step 2 — Compute cross-modal compatibility:**

$$
S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{M \times N}
$$

Entry $S_{ij} = q_i^\top k_j / \sqrt{d_k}$ measures how much text token $i$ attends to image patch $j$.

**Step 3 — Softmax over image patches (columns of $K$):**

$$
\alpha_{ij} = \frac{\exp(S_{ij})}{\sum_{l=1}^{N} \exp(S_{il})}, \quad \sum_{j=1}^{N} \alpha_{ij} = 1
$$

**Why:** Each text token distributes one unit of attention across all image regions.

**Step 4 — Aggregate image values:**

$$
\text{CrossAttn}(Z_{\text{txt}}, Z_{\text{img}})_i = \sum_{j=1}^{N} \alpha_{ij} v_j
$$

Matrix form: $\text{CrossAttn} = \text{softmax}(QK^\top/\sqrt{d_k}) V$. **∎**

#### Numerical Example

$M=2$ text tokens, $N=3$ patches, $d_k=2$. After softmax row 0: $\alpha_{0,:} = [0.422, 0.156, 0.422]$. Output for token 0: $0.422 v_1 + 0.156 v_2 + 0.422 v_3$ — equal weight on patches 1 and 3.

---

### Proof: Bilinear Fusion — Rank Analysis

**Claim:** Full bilinear fusion $z_k = \mathbf{v}^\top W_k \mathbf{t}$ has rank at most $\min(d_v, d_t)$ per output, and parameter count grows as $O(d_v d_t d_{\text{out}})$.

**Step 1 — Write bilinear form:**

$$
\mathbf{z} = \mathbf{v}^\top \mathcal{W} \mathbf{t}, \quad \mathcal{W} \in \mathbb{R}^{d_v \times d_t \times d_{\text{out}}}
$$

Each slice $W_k \in \mathbb{R}^{d_v \times d_t}$ is a rank-$\min(d_v, d_t)$ matrix at most.

**Step 2 — Rank bottleneck:**

If $d_v = d_t = 768$ and $d_{\text{out}} = 512$, we need 768×768×512 $\approx$ 302M parameters — most slices are redundant because finetuning updates lie in a low-rank subspace.

**Step 3 — Low-rank factorization (MLB):**

Factor $W_k \approx U_k V_k^\top$ with $U_k \in \mathbb{R}^{d_v \times r}$, $V_k \in \mathbb{R}^{d_t \times r}$:

$$
\mathbf{z} = (U \mathbf{v}) \odot (V \mathbf{t}), \quad \text{Params} = r(d_v + d_t)
$$

**Why this works:** Eckart–Young says the best rank-$r$ approximation captures most energy; MLB applies this per interaction. **∎**

#### Numerical Example

$d_v = d_t = 768$, $d_{\text{out}} = 512$, $r = 64$:

- Full bilinear: $768 \times 768 \times 512 = 301,989,888$ params
- MLB: $64 \times (768 + 768) = 98,304$ params → **3072× reduction**

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

## 🔬 Worked Examples in the Notebook

### Example 1: Early Fusion — Numerical Trace
Concatenate 2 image patches + 3 text tokens, then compute full 5×5 attention:
- See cross-modal attention patterns (does "txt:cat" attend to "img:cat"?)
- Measure cross-modal vs intra-modal attention strength

### Example 2: Cross-Attention — Complete Walkthrough
Trace cross-attention for "cat" querying 4 image patches:
- Q·K dot products → scaling → softmax → weighted Value sum
- "cat" correctly attends to cat-patch (highest weight)
- Full multi-query cross-attention heatmap

### Example 3: Bilinear/MLB Fusion Implementation
Implement full bilinear and low-rank MLB fusion:
- Compare parameter counts: Full (1M+) vs MLB rank-32 (12K)
- At ViT-Base scale: 452M vs 1.6M parameters (282x compression!)

### Example 4: Side-by-Side Training Comparison
Train concat, add, gated, and bilinear fusion on the SAME task:
- Loss and accuracy curves for 100 epochs
- Compare final accuracy and parameter efficiency
- Winner and most-efficient method identified

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/03_fusion_strategies/03_fusion_strategies.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| Early Fusion (VisualBERT) | Li et al. — [arXiv:1908.03557](https://arxiv.org/abs/1908.03557) | 2019 | Concatenate vision + text tokens → joint transformer |
| Co-Attention (ViLBERT) | Lu et al. — [arXiv:1908.02265](https://arxiv.org/abs/1908.02265) | 2019 | Dual-stream transformers with K,V exchange |
| Cross-Attention Bridge (Perceiver/BLIP-2) | Jaegle et al. / Li et al. — [arXiv:2103.03206](https://arxiv.org/abs/2103.03206) | 2021-23 | Learnable queries + cross-attention to frozen encoders |

---

## Next Step

**[01_clip_from_scratch](../../02_Vision_Language_Models/01_clip_from_scratch/)** — Build CLIP end-to-end
