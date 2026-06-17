# 02 — Modality Encoders: Image & Text

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/02_modality_encoders/02_modality_encoders.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

How the two most important encoders work — **from first principles**:

1. **Vision Transformer (ViT)** — how images become sequences of vectors
2. **Text Encoder (BERT-style)** — how tokens become contextualized embeddings

---

## Part 1: Vision Transformer (ViT)

### Complete Pipeline — Tensor Shapes at Every Step

```
  Input Image: x ∈ ℝ^(3 × 224 × 224)
       │
       ▼  Step 1: Patch extraction (P = 16)
  ┌────────────────────────────────────────────────┐
  │  Split into non-overlapping 16×16 patches      │
  │  Number of patches: N = (224/16)² = 196        │
  │  Each patch: p_i ∈ ℝ^(3 × 16 × 16) = ℝ^768   │
  └────────────────────┬───────────────────────────┘
                       │
       ▼  Step 2: Linear projection (patch embedding)
  ┌────────────────────────────────────────────────┐
  │  E ∈ ℝ^(768 × D)   where D = model dim        │
  │  z_i = Flatten(p_i) · E + b                    │
  │  Output: [z₁, z₂, ..., z₁₉₆]  each ∈ ℝ^D     │
  └────────────────────┬───────────────────────────┘
                       │
       ▼  Step 3: Prepend [CLS] token + positional encoding
  ┌────────────────────────────────────────────────┐
  │  z₀ = z_CLS  (learnable, ∈ ℝ^D)               │
  │  Z = [z₀, z₁, ..., z₁₉₆] + PE                │
  │  Z ∈ ℝ^(197 × D)                              │
  └────────────────────┬───────────────────────────┘
                       │
       ▼  Step 4: L Transformer layers
  ┌────────────────────────────────────────────────┐
  │  For each layer l = 1, ..., L:                 │
  │    Z' = Z + MSA(LN(Z))                         │
  │    Z  = Z' + FFN(LN(Z'))                       │
  │  Output: [h₀, h₁, ..., h₁₉₆]  each ∈ ℝ^D     │
  └────────────────────┬───────────────────────────┘
                       │
       ▼  Step 5: Take [CLS] as image representation
  v = h₀ ∈ ℝ^D        (global image embedding)
```

### ViT Variants — Parameter Comparison

| Variant | Layers $L$ | Dim $D$ | Heads $h$ | Params | Patches |
|---------|-----------|---------|----------|--------|---------|
| ViT-Tiny | 12 | 192 | 3 | 5.7M | 196 |
| ViT-Small | 12 | 384 | 6 | 22M | 196 |
| ViT-Base | 12 | 768 | 12 | 86M | 196 |
| ViT-Large | 24 | 1024 | 16 | 307M | 196 |
| ViT-Huge | 32 | 1280 | 16 | 632M | 196 |

---

## Key Equations — Transformer Building Blocks

### 1. Self-Attention (The Heart of Transformers)

Given input $Z \in \mathbb{R}^{n \times D}$, we project into queries, keys, and values:

$$
Q = Z W_Q, \quad K = Z W_K, \quad V = Z W_V
$$

where $W_Q, W_K \in \mathbb{R}^{D \times d_k}$ and $W_V \in \mathbb{R}^{D \times d_v}$.

The attention output:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**Why $\sqrt{d_k}$ scaling?** Without scaling, the dot products $QK^\top$ grow proportionally to $d_k$. If entries of $Q$ and $K$ are i.i.d. with mean 0 and variance 1:

$$
\text{Var}(q^\top k) = \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k \cdot \text{Var}(q_i)\text{Var}(k_i) = d_k
$$

Dividing by $\sqrt{d_k}$ restores unit variance, preventing softmax saturation:

$$
\text{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

### 2. Multi-Head Attention (MHA)

Instead of one attention function, run $h$ parallel heads with different projections:

$$
\text{head}_i = \text{Attention}(Z W_Q^{(i)}, Z W_K^{(i)}, Z W_V^{(i)})
$$

$$
\text{MultiHead}(Z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

where each head operates on $d_k = d_v = D/h$ dimensions.

**Parameter count per MHA layer:**

$$
\text{Params}_{\text{MHA}} = 4 D^2 \quad (W_Q, W_K, W_V, W^O \text{ each } D \times D)
$$

**Numerical example** (ViT-Base: $D = 768, h = 12$):

$$
d_k = d_v = 768/12 = 64 \text{ per head}
$$

$$
\text{Params}_{\text{MHA}} = 4 \times 768^2 = 2{,}359{,}296 \approx 2.4\text{M}
$$

### 3. Layer Normalization

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i, \quad \sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2
$$

$\gamma, \beta \in \mathbb{R}^D$ are learnable (element-wise scale and shift), $\epsilon \approx 10^{-6}$.

**Why LN instead of BatchNorm?** LN normalizes across features (dim $D$) per token, making it independent of batch size — critical for variable-length sequences and small batches.

### 4. Feed-Forward Network (FFN)

Each transformer layer has a position-wise FFN with expansion ratio (typically 4×):

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

where $W_1 \in \mathbb{R}^{D \times 4D}$, $W_2 \in \mathbb{R}^{4D \times D}$.

**GELU activation:**

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

**Parameter count per FFN:**

$$
\text{Params}_{\text{FFN}} = 2 \times D \times 4D = 8D^2
$$

### 5. Positional Encoding

ViT uses **learnable** positional embeddings $\text{PE} \in \mathbb{R}^{(N+1) \times D}$, one vector per position (including [CLS]).

For reference, the original Transformer used **sinusoidal** encoding:

$$
\text{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/D}}\right), \quad \text{PE}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/D}}\right)
$$

**Why positions matter:** Without PE, the Transformer is permutation-invariant — it can't distinguish patch order. Positional encoding breaks this symmetry.

### 6. Residual Connections (Per Transformer Block)

$$
Z' = Z + \text{MSA}(\text{LN}(Z))
$$

$$
Z_{\text{out}} = Z' + \text{FFN}(\text{LN}(Z'))
$$

This is **Pre-LN** (used in ViT), where LayerNorm is applied **before** attention/FFN. Pre-LN stabilizes training and enables learning rate warmup to be shorter.

---

## Mathematical Proofs

### Proof: Self-Attention — Why Q, K, V and $\text{softmax}(QK^\top/\sqrt{d_k})V$?

**Claim:** Attention computes a convex combination of value vectors, where weights come from query–key compatibility.

**Step 1 — Define projections from input $Z \in \mathbb{R}^{n \times D}$:**

$$
Q = Z W_Q, \quad K = Z W_K, \quad V = Z W_V
$$

**Why:** $W_Q, W_K, W_V$ are learnable linear maps that let the model decide *what to look for* (Q), *what to match against* (K), and *what to retrieve* (V).

**Step 2 — Compute compatibility scores:**

For query token $i$ and key token $j$, the score is the dot product $q_i^\top k_j$. Stacking all pairs gives $\mathbf{S} = QK^\top \in \mathbb{R}^{n \times n}$.

**Why:** Dot products measure directional similarity in the projected subspace — higher score means "more relevant."

**Step 3 — Normalize scores to a probability distribution:**

$$
\alpha_{ij} = \frac{\exp(S_{ij}/\sqrt{d_k})}{\sum_{l=1}^{n} \exp(S_{il}/\sqrt{d_k})} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)_{ij}
$$

**Why:** Softmax ensures $\sum_j \alpha_{ij} = 1$, so each output is a weighted average of values.

**Step 4 — Aggregate values:**

$$
\text{output}_i = \sum_{j=1}^{n} \alpha_{ij} v_j = \left(\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V\right)_i
$$

**∎**

#### Numerical Example

3 tokens, $d_k = 2$, $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$, $K = Q$, $V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$.

Scores for token 0: $QK^\top/\sqrt{2} = [0.707, 0, 0.707]$. Softmax $\approx [0.422, 0.156, 0.422]$. Output $\approx 0.422[1,0] + 0.156[0,1] + 0.422[1,1] = [0.844, 0.578]$.

---

### Proof: Why Scale by $\sqrt{d_k}$?

**Claim:** For random vectors $q, k \in \mathbb{R}^{d_k}$ with components $\sim \mathcal{N}(0, 1)$:

$$
\text{Var}(q \cdot k) = d_k
$$

**Proof:**

**Step 1:** The dot product is $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$.

**Step 2:** Each $q_i k_i$ has mean 0 and variance:

$$
\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = 1 \cdot 1 = 1
$$

**Step 3:** Since the $d_k$ terms are independent:

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
$$

**Step 4:** Dividing by $\sqrt{d_k}$ gives:

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

This keeps the softmax inputs in a reasonable range, preventing gradient vanishing. **∎**

#### Numerical Verification

For $d_k = 64$: unscaled std $\approx \sqrt{64} = 8$, scaled std $\approx 1$.

With scores $[8.2, -7.1, 6.5]$: softmax $\to [0.847, 0.000, 0.153]$ (almost one-hot!)

With scores $[1.02, -0.89, 0.81]$: softmax $\to [0.436, 0.065, 0.354]$ (smooth distribution ✓)

---

## Part 2: Text Encoder (BERT-style)

### Pipeline

```
  Input: "A cat sitting on a mat"
       │
       ▼  Tokenization (WordPiece / BPE)
  [CLS] "a" "cat" "sitting" "on" "a" "mat" [SEP]
  ids = [101, 1037, 4937, 3564, 2006, 1037, 13523, 102]
       │
       ▼  Token embedding + Position embedding + Segment embedding
  E = TokenEmb(ids) + PosEmb(0..7) + SegEmb(0)
  E ∈ ℝ^(8 × D)
       │
       ▼  L Transformer layers (bidirectional attention)
  H = TransformerEncoder(E)
  H ∈ ℝ^(8 × D)
       │
       ▼  Take [CLS] token as text representation
  t = H[0] ∈ ℝ^D
```

**Key difference from ViT:** BERT uses **bidirectional** attention (every token attends to every other token), while GPT-style decoders use **causal** (left-to-right) attention.

### Attention Mask Comparison

```
  BERT (Bidirectional):        GPT (Causal):
  ┌─────────────┐              ┌─────────────┐
  │ 1  1  1  1  │              │ 1  0  0  0  │
  │ 1  1  1  1  │              │ 1  1  0  0  │
  │ 1  1  1  1  │              │ 1  1  1  0  │
  │ 1  1  1  1  │              │ 1  1  1  1  │
  └─────────────┘              └─────────────┘
  Every token sees all.        Each token sees only past.
```

---

## Total Parameters in ViT-Base (Worked Example)

| Component | Formula | Params |
|-----------|---------|--------|
| Patch embedding | $3 \times 16^2 \times 768$ | 590K |
| [CLS] token | $768$ | 768 |
| Position embedding | $197 \times 768$ | 151K |
| Per-layer MHA | $4 \times 768^2$ | 2.36M |
| Per-layer FFN | $8 \times 768^2$ | 4.72M |
| Per-layer LN ($\times 2$) | $2 \times 2 \times 768$ | 3.1K |
| **Per layer total** | | **7.08M** |
| **12 layers** | $12 \times 7.08\text{M}$ | **85M** |
| **Grand total** | + embeddings | **~86M** |

---

## What You'll Build

- ViT image encoder from scratch (patch extraction → transformer → CLS pooling)
- BERT-style text encoder from scratch
- Visualization of attention patterns and learned representations
- Parameter counting verification

---

## Prerequisites

- Notebook 01 (cosine similarity, InfoNCE basics)
- Matrix multiplication and tensor reshaping
- Basic understanding of convolutional vs. attention-based encoders

---

## 🔬 Worked Examples in the Notebook

### Example 1: Self-Attention — Step-by-Step with Real Numbers
Trace attention for 3 tokens ("cat", "sits", "mat") with $d_k=4$:
- Compute Q, K, V from input X and weight matrices
- Raw scores → scaling by $\sqrt{d_k}$ → softmax → weighted sum
- Visualize: attention heatmap, scaling effect, temperature sensitivity

### Example 2: ViT-Base Memory & Compute Budget
Calculate exact resource requirements for ViT variants:
- Parameters: Patch embedding → Position embedding → 12 Transformer layers
- Memory: FP32/FP16 weights + activations + attention matrices
- FLOPs: 17.4 GFLOPs per image for ViT-Base
- Throughput estimates: A100 (~7,000 img/s), T4 (~2,000 img/s)
- Compare ViT-Tiny through ViT-Large

### Example 3: Simulated Product Search Pipeline
Build a complete image-text retrieval system:
- Encode 100 products with ViT + text encoder
- Image-to-image retrieval: find similar products
- Text-to-image retrieval: search by description
- Visualize similarity distributions (same vs different category)

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/02_modality_encoders/02_modality_encoders.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| ViT Architecture (`../../assets/paper_figures/vit_architecture.png`) | Dosovitskiy et al. — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | 2020 | Official Google ViT figure: patches → Transformer |
| ViT Architecture | Dosovitskiy et al. — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | 2020 | Image → patches → Transformer → [CLS] output |

### Additional Papers Covered

- **DeiT** (Touvron et al., 2021) — Data-efficient image transformers with distillation
- **BEiT** (Bao et al., 2021) — BERT pre-training for image transformers
- **BERT** (Devlin et al., 2019) — Bidirectional encoder representations from transformers

---

## Next Step

**[03_fusion_strategies](../03_fusion_strategies/)** — How to combine image and text representations
