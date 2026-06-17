# 03 — Adapter Methods: Beyond LoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/03_adapter_methods/03_adapter_methods.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

LoRA isn't the only PEFT method. This notebook covers **5 alternative approaches**, their math, and when to use each:

| Method | Where | Trainable Params | Inference Overhead | Mergeable? |
|--------|-------|-----------------|-------------------|-----------|
| **Bottleneck Adapter** | Between layers | ~2% | Yes (extra layers) | No |
| **Prefix Tuning** | KV prepend | ~0.1% | Yes (extra tokens) | No |
| **Prompt Tuning** | Input prepend | ~0.01% | Yes (extra tokens) | No |
| **IA3** | Scale K, V, FFN | ~0.01% | Mergeable | Yes |
| **BitFit** | Bias terms only | ~0.05% | None | Yes (built-in) |

---

## Method 1: Bottleneck Adapters

### Architecture

Insert a small **bottleneck MLP** after each transformer sub-layer:

```
  ┌──────────────────────────────────┐
  │ Transformer Layer                 │
  │                                   │
  │  Input x                          │
  │    │                              │
  │    ▼                              │
  │  [Self-Attention] (frozen)        │
  │    │                              │
  │    ▼                              │
  │  ┌─────────────────────┐         │
  │  │  Adapter Module      │         │
  │  │  ┌────────────────┐ │         │
  │  │  │ Down-proj (d→r) │ │         │
  │  │  │ ReLU / GELU    │ │ trainable│
  │  │  │ Up-proj (r→d)  │ │         │
  │  │  └───────┬────────┘ │         │
  │  │          │ + skip    │         │
  │  │          ▼           │         │
  │  └─────────────────────┘         │
  │    │                              │
  │    ▼                              │
  │  [FFN] (frozen)                   │
  │    │                              │
  │    ▼                              │
  │  [Adapter Module] (same structure)│
  │    │                              │
  │    ▼                              │
  │  Output                           │
  └──────────────────────────────────┘
```

### Mathematical Formulation

$$
\text{Adapter}(x) = x + f(x W_{\text{down}}) W_{\text{up}}
$$

where:
- $W_{\text{down}} \in \mathbb{R}^{d \times r}$ — project down to bottleneck ($r \ll d$)
- $f$ — non-linear activation (ReLU or GELU)
- $W_{\text{up}} \in \mathbb{R}^{r \times d}$ — project back up
- $+ x$ — residual (skip) connection

**Parameter count per adapter:** $2 \times d \times r + r + d$ (weights + biases)

**For ViT-Base ($d = 768$, $r = 64$, 2 adapters per layer, 12 layers):**

$$
12 \times 2 \times (2 \times 768 \times 64 + 64 + 768) = 2{,}377{,}344 \approx 2.4\text{M} \; (2.8\% \text{ of 86M})
$$

### Initialization

$$
W_{\text{down}} \sim \mathcal{N}(0, 0.01^2), \quad W_{\text{up}} \sim \mathcal{N}(0, 0.01^2)
$$

Near-zero initialization ensures the adapter starts as an approximate identity function.

---

## Method 2: Prefix Tuning

### Idea

Prepend **learnable key-value pairs** to the attention at every layer:

```
  Normal attention:
  K = [K_x₁, K_x₂, ..., K_xₙ]         (n tokens)
  V = [V_x₁, V_x₂, ..., V_xₙ]

  With prefix:
  K = [P_K₁, ..., P_Kₘ, K_x₁, ..., K_xₙ]   (m + n tokens)
  V = [P_V₁, ..., P_Vₘ, V_x₁, ..., V_xₙ]

  where P_K, P_V ∈ ℝ^(m × d) are LEARNED
```

### Mathematical Formulation

$$
\text{Attention}(Q, [P_K; K_x], [P_V; V_x]) = \text{softmax}\!\left(\frac{Q [P_K; K_x]^\top}{\sqrt{d_k}}\right) [P_V; V_x]
$$

Expanding the softmax:

$$
\alpha_i = \text{softmax}\!\left(\frac{q_i^\top [p_{k_1}, \ldots, p_{k_m}, k_{x_1}, \ldots, k_{x_n}]}{\sqrt{d_k}}\right)
$$

The first $m$ attention weights attend to the **learned prefix**, the remaining $n$ attend to the actual input. The prefix acts as a "steering signal" that biases attention toward task-relevant features.

### Reparameterization (Training Stability)

Directly optimizing $P_K, P_V$ is unstable. Instead, use an MLP to generate them:

$$
P_K^{(l)} = \text{MLP}_K(E_K^{(l)}), \quad P_V^{(l)} = \text{MLP}_V(E_V^{(l)})
$$

where $E_K, E_V \in \mathbb{R}^{m \times d'}$ are the actual learned parameters, and the MLP maps $d' \to d$.

**Parameter count ($m = 20$, $L = 12$ layers, $d = 768$):**

$$
\text{Trainable} = L \times m \times 2 \times d = 12 \times 20 \times 2 \times 768 = 368{,}640 \; (0.43\%)
$$

---

## Method 3: Prompt Tuning

### Idea — Simplest PEFT Method

Prepend $m$ learnable token embeddings to the **input** (only at the first layer):

$$
Z_{\text{input}} = [P_1, P_2, \ldots, P_m, x_1, x_2, \ldots, x_n]
$$

where $P_i \in \mathbb{R}^d$ are learned "soft prompts."

```
  Hard prompt (text):      "Classify this image:"
  Soft prompt (learned):   [P₁, P₂, ..., P₂₀]  ← continuous vectors, not words!

  Combined input:
  [P₁, P₂, ..., P₂₀, x₁, x₂, ..., xₙ]
       └──── learned ────┘  └── input ──┘
```

**Parameter count ($m = 20$, $d = 768$):**

$$
\text{Trainable} = m \times d = 20 \times 768 = 15{,}360 \; (0.018\%)
$$

### Comparison with Prefix Tuning

| | Prompt Tuning | Prefix Tuning |
|---|-------------|---------------|
| Applied at | Input layer only | Every layer's KV |
| Parameters | $m \times d$ | $L \times m \times 2d$ |
| Interaction | Through natural attention propagation | Direct steering at each layer |
| Typical $m$ | 20–100 | 10–30 |
| Effectiveness | Works well for large models (>10B) | Better for smaller models |

---

## Method 4: IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

### Idea — Element-wise Rescaling

Instead of adding new modules, learn **per-dimension scaling vectors** for Keys, Values, and FFN:

$$
K' = l_K \odot K, \quad V' = l_V \odot V, \quad h_{\text{FFN}}' = l_{\text{FFN}} \odot h_{\text{FFN}}
$$

where $l_K, l_V, l_{\text{FFN}} \in \mathbb{R}^d$ are learned vectors (initialized to $\mathbf{1}$).

```
  Standard:                    With IA3:
  ┌──────────────────┐         ┌──────────────────┐
  │ K = X · W_K      │         │ K = l_K ⊙ (X·W_K) │
  │ V = X · W_V      │         │ V = l_V ⊙ (X·W_V) │
  │                   │         │                    │
  │ FFN = W₂·σ(W₁·x) │         │ FFN = l_FFN ⊙ W₂·σ(W₁·x) │
  └──────────────────┘         └──────────────────┘

  l_K, l_V, l_FFN ∈ ℝᵈ  (just d-dimensional vectors!)
```

**Parameter count (ViT-Base: $d = 768$, 12 layers):**

$$
\text{Trainable} = L \times 3d = 12 \times 3 \times 768 = 27{,}648 \; (0.032\%)
$$

**Key advantage:** Like LoRA, IA3 can be **merged** at inference time:

$$
W_K^{\text{merged}} = \text{diag}(l_K) \cdot W_K
$$

No inference overhead.

---

## Method 5: BitFit — Bias-Only Finetuning

### Idea

Only finetune the **bias terms** of the model. All weight matrices remain frozen:

$$
\text{Finetune:} \quad \{b_Q, b_K, b_V, b_O, b_1, b_2, b_{\text{LN}}\}
$$

$$
\text{Freeze:} \quad \{W_Q, W_K, W_V, W_O, W_1, W_2, \gamma_{\text{LN}}\}
$$

**Parameter count (ViT-Base):**

$$
\text{Trainable} = L \times (4d + 4d + d + 2d) = 12 \times 11 \times 768 = 101{,}376 \; (0.12\%)
$$

**When it works:** Surprisingly effective for tasks close to the pre-training distribution. Less effective for large domain shifts.

---

## Comprehensive Comparison

### Architecture Diagrams

```
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Adapter      │  │  LoRA         │  │  IA3          │
  │               │  │               │  │               │
  │  x → Down →  │  │  x → A →     │  │  K' = l_K⊙K  │
  │  ReLU → Up → │  │  B → (α/r)·  │  │  V' = l_V⊙V  │
  │  + skip       │  │  + W₀x       │  │  h' = l_f⊙h  │
  │               │  │               │  │               │
  │  Params: 2dr  │  │  Params: 2dr  │  │  Params: 3d   │
  │  Mergeable: ✗ │  │  Mergeable: ✓ │  │  Mergeable: ✓ │
  └──────────────┘  └──────────────┘  └──────────────┘

  ┌──────────────┐  ┌──────────────┐
  │  Prefix       │  │  BitFit       │
  │               │  │               │
  │  K=[P_K; K_x] │  │  Only update  │
  │  V=[P_V; V_x] │  │  bias terms   │
  │               │  │  b_Q, b_K, ...│
  │  Params: L·2md│  │               │
  │  Mergeable: ✗ │  │  Params: ~11d │
  └──────────────┘  │  Mergeable: ✓ │
                    └──────────────┘
```

### Quantitative Comparison (ViT-Base, 86M params)

| Method | Trainable | % | Inference Overhead | Mergeable | Best For |
|--------|-----------|---|-------------------|-----------|----------|
| Full FT | 86M | 100% | None | N/A | Unlimited compute |
| LoRA ($r=8$) | 147K | 0.17% | None (merged) | Yes | General-purpose PEFT |
| Adapter ($r=64$) | 2.4M | 2.8% | +5% latency | No | When LoRA isn't enough |
| Prefix ($m=20$) | 369K | 0.43% | +20 tokens/layer | No | Few-shot, NLG tasks |
| Prompt ($m=20$) | 15K | 0.018% | +20 tokens | No | Very large models |
| IA3 | 28K | 0.032% | None (merged) | Yes | Extreme parameter budget |
| BitFit | 101K | 0.12% | None | Yes | Small domain shift |

### Decision Guide

```
  How much compute do you have?
  │
  ├── Unlimited → Full Finetuning
  │
  ├── 1 GPU (16-80GB) → LoRA (best default)
  │
  ├── 1 GPU (4-8GB) → QLoRA (quantized + LoRA)
  │
  └── Extreme constraint → Which is more important?
                            │
                            ├── Accuracy → IA3 or Adapter
                            │
                            └── Simplicity → BitFit or Prompt Tuning
```

---

## Mathematical Proofs

### Proof: Bottleneck Adapter as Information Bottleneck

**Claim:** The down-up adapter $\text{Adapter}(x) = x + f(x W_{\text{down}}) W_{\text{up}}$ compresses information through rank-$r$ bottleneck.

**Step 1 — Information flow:**

$$
x \in \mathbb{R}^d \xrightarrow{W_{\text{down}}} h \in \mathbb{R}^r \xrightarrow{f, W_{\text{up}}} \Delta x \in \mathbb{R}^d
$$

**Step 2 — Bottleneck constraint:** With $r \ll d$, the adapter can only pass $r$ degrees of freedom — forcing compression of task-specific signal.

**Step 3 — Information bottleneck principle:** Minimize $\mathcal{L}_{\text{task}}$ while limiting $I(x; h)$ — low-rank bottleneck approximates this by restricting channel capacity to $r$ dimensions.

**Step 4 — Residual preserves pretrained knowledge:** $x + \Delta x$ keeps base representation; adapter adds task-specific delta in low-rank subspace. **∎**

#### Numerical Example

$d=768$, $r=64$: bottleneck compresses 768-dim activation to 64-dim — 12× compression. Adapter params $= 2 \times 768 \times 64 = 98{,}304$ vs full layer $768^2 = 589{,}824$.

---

### Proof: Prefix Tuning Equivalence to Soft Prompts in Attention

**Claim:** Prefix tuning prepending $P_K, P_V$ to attention is equivalent to adding learned soft tokens that steer attention at every layer.

**Step 1 — Augmented keys and values:**

$$
K' = [P_K; K_x], \quad V' = [P_V; V_x]
$$

**Step 2 — Attention with prefix:**

$$
\text{Attn} = \text{softmax}\!\left(\frac{Q [P_K; K_x]^\top}{\sqrt{d_k}}\right) [P_V; V_x]
$$

**Step 3 — Soft prompt interpretation:** The first $m$ positions attend to learned virtual tokens $P_K, P_V$ — not derived from input — analogous to prompt tuning's $P_i \in \mathbb{R}^d$ but applied at every layer's KV.

**Step 4 — Expressiveness:** Prefix tuning ($L \times m \times 2d$ params) > prompt tuning ($m \times d$ params) because it steers each layer directly rather than relying on propagation through depth. **∎**

#### Numerical Example

$m=20$, $L=12$, $d=768$: prefix params $= 12 \times 20 \times 2 \times 768 = 368{,}640$. Prompt tuning: $20 \times 768 = 15{,}360$ — prefix has 24× more capacity for layer-wise control.

---

## What You'll Build

- All 5 PEFT methods implemented from scratch
- Side-by-side comparison on the same task
- Parameter count comparison chart
- Training speed and memory benchmarks
- Accuracy vs. parameter efficiency Pareto frontier

---

## Prerequisites

- Notebook 01 (LoRA from scratch)
- Understanding of attention mechanisms (Q, K, V)
- Familiarity with PyTorch hooks and parameter freezing

---

## 🔬 Worked Examples in the Notebook

### PEFT Method Comparison — Training on Same Task
- Train Full FT, LoRA (r=8), Prompt Tuning, and Head Only on same data
- 60 epochs, 8-class classification with 400 samples
- Loss curves, final accuracy, and parameter counts compared
- Identify best accuracy/efficiency trade-off

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/03_adapter_methods/03_adapter_methods.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| PEFT Methods Comparison (4-panel) | Houlsby / Li & Liang / Lester / Hu | 2019-21 | Where trainable parameters are inserted in each method |

### Papers Referenced

- **Bottleneck Adapter** (Houlsby et al., 2019) — Inserted after attention + FFN
- **Prefix Tuning** (Li & Liang, 2021) — Learnable K,V tokens at each attention layer
- **Prompt Tuning** (Lester et al., 2021) — Soft prompt tokens at input only
- **IA³** (Liu et al., 2022) — Learned rescaling vectors, fewest parameters

---

## Next Step

**[04_finetune_clip_custom_data](../04_finetune_clip_custom_data/)** — Practical capstone: finetune CLIP on your own data
