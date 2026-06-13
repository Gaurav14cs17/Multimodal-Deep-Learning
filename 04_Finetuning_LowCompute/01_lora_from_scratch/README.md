# 01 — LoRA from Scratch: Build It, Understand It

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/01_lora_from_scratch/01_lora_from_scratch.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

**LoRA (Low-Rank Adaptation)** is THE technique for finetuning large models with limited compute.

- Why low-rank works — intrinsic dimensionality of weight updates
- SVD and the Eckart-Young theorem (with worked example)
- Build a LoRA layer from scratch in PyTorch
- Compare trainable parameters: full FT vs LoRA
- Apply LoRA to a transformer and train it

---

## Core Idea — Low-Rank Weight Updates

During finetuning, the weight update $\Delta W$ has **low intrinsic rank**. Instead of learning all $d \times d$ parameters, LoRA decomposes the update into two small matrices:

$$
h = W_0 x + \frac{\alpha}{r} \underbrace{B A}_{d \times r \;\cdot\; r \times d} x
$$

where:
- $W_0 \in \mathbb{R}^{d \times d}$ is the **frozen** pretrained weight
- $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ are **trainable** low-rank matrices
- $r \ll d$ is the rank (typically $r \in \{4, 8, 16, 32, 64\}$)
- $\alpha$ is a scaling factor (controls the magnitude of the update)

```
  ┌────────────────────────────────────────────────────────────┐
  │  Full Finetuning:              LoRA:                       │
  │  ┌───────────┐                 ┌───────────┐               │
  │  │  W + ΔW   │                 │  W (frozen)│──► W₀x       │
  │  │  (all d²) │                 │            │     +        │
  │  └───────────┘                 └───────────┘   ┌───┐       │
  │                                   x ──────────►│ A │ r×d   │
  │  Parameters: d²                                │   │       │
  │  = 7.7M (d=768)                                └─┬─┘       │
  │                                                  │         │
  │                                                ┌─▼─┐       │
  │                                                │ B │ d×r   │
  │                                                │   │       │
  │                                                └─┬─┘       │
  │                                     (α/r)·BAx ──┘         │
  │                                                            │
  │                                Parameters: 2·d·r           │
  │                                = 12,288 (r=8, d=768)       │
  │                                → 99.8% reduction           │
  └────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundation — SVD and Low-Rank Approximation

### Singular Value Decomposition (SVD)

Any matrix $W \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$
W = U \Sigma V^\top = \sum_{i=1}^{\min(m,n)} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

where:
- $U \in \mathbb{R}^{m \times m}$ — left singular vectors (orthonormal)
- $\Sigma \in \mathbb{R}^{m \times n}$ — diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$
- $V \in \mathbb{R}^{n \times n}$ — right singular vectors (orthonormal)

### Eckart-Young Theorem — Optimal Low-Rank Approximation

The best rank-$r$ approximation of $W$ (minimizing Frobenius norm error) is:

$$
W_r = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_r \Sigma_r V_r^\top
$$

The approximation error:

$$
\lVert W - W_r \rVert_F^2 = \sum_{i=r+1}^{\min(m,n)} \sigma_i^2
$$

### Worked Numerical Example

Consider a $4 \times 4$ weight matrix:

$$
W = \begin{pmatrix} 2.1 & 0.9 & 1.5 & 0.3 \\ 1.8 & 0.8 & 1.3 & 0.3 \\ 4.2 & 1.8 & 3.0 & 0.6 \\ 0.6 & 0.3 & 0.5 & 0.1 \end{pmatrix}
$$

SVD gives singular values: $\sigma_1 = 6.5, \sigma_2 = 0.4, \sigma_3 = 0.02, \sigma_4 = 0.001$

**Energy captured by top-$r$ components:**

| Rank $r$ | Energy $\sum_{i=1}^r \sigma_i^2 / \sum \sigma_i^2$ | Relative Error |
|----------|------------------------------------------------|---------------|
| 1 | 99.6% | 0.4% |
| 2 | 99.99% | 0.01% |
| 3 | 99.999% | 0.001% |
| 4 | 100% | 0% |

**Observation:** Rank 1 captures 99.6% of the energy! This is typical for neural network weight matrices — they are approximately low-rank.

### Why Does This Work for Finetuning?

Research (Aghajanyan et al., 2021) showed that during finetuning, the weight change $\Delta W = W_{\text{finetuned}} - W_{\text{pretrained}}$ has **low intrinsic dimensionality**:

$$
d_{\text{intrinsic}} \ll d^2
$$

For GPT-3 (175B params), $d_{\text{intrinsic}} \approx 5{,}000$ — meaning only $\sim 0.003\%$ of parameters actually need to change.

---

## LoRA Initialization

Critical for stable training:

$$
A \sim \mathcal{N}(0, \sigma^2), \quad B = \mathbf{0}
$$

**Why $B = 0$?** At initialization, $\Delta W = BA = \mathbf{0} \cdot A = \mathbf{0}$, so the model starts **exactly** at the pretrained checkpoint. No training instability from random initialization.

**Scaling factor $\alpha/r$:**

$$
\Delta W = \frac{\alpha}{r} BA
$$

When you change $r$, the scaling $\alpha/r$ keeps the magnitude of the update roughly constant:

| $r$ | $\alpha$ | $\alpha/r$ | Effect |
|-----|---------|-----------|--------|
| 4 | 8 | 2.0 | Stronger update per rank |
| 8 | 16 | 2.0 | Same effective scale |
| 16 | 32 | 2.0 | Same effective scale |
| 64 | 64 | 1.0 | Weaker per rank, more capacity |

---

## Which Layers to Apply LoRA?

### Standard Practice

| Target | Applied To | Why |
|--------|-----------|-----|
| $W_Q, W_V$ | Query and Value projections | Default CLIP/LLM setting, best quality/parameter ratio |
| $W_Q, W_K, W_V, W_O$ | All attention projections | Better for complex tasks |
| + $W_{\text{up}}, W_{\text{down}}$ | + FFN layers | Most parameters, best for domain shift |
| All linear layers | Everything | Maximum capacity, still fewer params than full FT |

### Parameter Count Comparison (ViT-Base, $D = 768$, 12 layers)

| Strategy | Trainable | % of Total | Formula |
|----------|-----------|-----------|---------|
| Full FT | 86M | 100% | All parameters |
| LoRA ($r=8$, Q+V) | 147K | 0.17% | $12 \times 2 \times 2 \times 768 \times 8$ |
| LoRA ($r=8$, Q+K+V+O) | 295K | 0.34% | $12 \times 4 \times 2 \times 768 \times 8$ |
| LoRA ($r=16$, all linear) | 1.77M | 2.1% | $12 \times 6 \times 2 \times 768 \times 16$ |

---

## LoRA Merging — Zero Inference Overhead

After training, merge LoRA weights back into the base model:

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA
$$

```
  Training time:                    Inference time:
  ┌────────┐   ┌───┐              ┌──────────────────┐
  │W₀(frozen)│   │LoRA│              │ W_merged          │
  │        │ + │(A,B)│    ──►     │ = W₀ + (α/r)·BA  │
  │        │   │    │              │                    │
  └────────┘   └───┘              └──────────────────┘
  2 forward passes                 1 forward pass
  (base + adapter)                 (no overhead!)
```

**This is why LoRA is preferred over Adapters/Prefix Tuning** — zero additional latency at inference time.

---

## Key Equations Summary

| Concept | Formula |
|---------|---------|
| LoRA forward | $h = W_0 x + \frac{\alpha}{r} BAx$ |
| Parameter savings | $1 - \frac{2r}{d} \approx 98\%$ (for $r=8$, $d=768$) |
| SVD | $W = U \Sigma V^\top$, keep top-$r$ singular values |
| Eckart-Young error | $\lVert W - W_r \rVert_F^2 = \sum_{i=r+1}^{k} \sigma_i^2$ |
| Initialization | $A \sim \mathcal{N}(0, \sigma^2)$, $B = \mathbf{0}$ |
| Merging | $W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$ (zero inference overhead) |

---

## What You'll Build

- LoRA layer from scratch (`LoRALinear` class in PyTorch)
- SVD analysis of pretrained weight matrices
- Full finetuning vs LoRA comparison experiment
- Parameter efficiency visualization
- LoRA merging and inference speed comparison

---

## Prerequisites

- Module 01–02 notebooks (Transformer architecture)
- Understanding of matrix multiplication and linear layers
- Basic PyTorch (`nn.Module`, `nn.Linear`)

---

## Next Step

**[02_qlora_4bit_finetuning](../02_qlora_4bit_finetuning/)** — Quantize to 4-bit + LoRA for extreme efficiency
