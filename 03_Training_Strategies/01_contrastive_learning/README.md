# 01 — Contrastive Learning Deep Dive

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning/01_contrastive_learning.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Intermediate–Advanced | **GPU Required:** No

---

## What You'll Learn

Contrastive learning is the **#1 training strategy** for multimodal models. This notebook goes deep:

- InfoNCE loss — full symmetric CLIP formula with step-by-step numerical example
- Connection to mutual information ($I(V;T) \geq \log N - \mathcal{L}$)
- Temperature parameter — variance analysis, gradient behavior, learnable $\tau$
- SigLIP — softmax-free alternative
- Hard negatives — probability analysis and why large batch sizes matter
- Training dynamics — watch the similarity matrix evolve from random to diagonal

---

## Key Equations

### 1. Symmetric InfoNCE (CLIP Loss)

Given $N$ image-text pairs, let $S_{ij} = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j$ be the cosine similarity:

$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} + \log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}\right]
$$

**What each term does:**
- First term (image→text): for each image $i$, classify which of the $N$ texts matches
- Second term (text→image): for each text $i$, classify which of the $N$ images matches
- Average of both directions ensures symmetric alignment

### 2. Mutual Information Lower Bound

InfoNCE provides a **lower bound** on the mutual information between modalities:

$$
I(V; T) \geq \log N - \mathcal{L}_{\text{InfoNCE}}
$$

**Derivation sketch:** The InfoNCE objective is equivalent to an $(N-1)$-way classification problem. A perfect classifier achieves $\mathcal{L} = 0$, giving $I(V; T) \geq \log N$. The tighter the bound, the more information the representations capture about each other.

**Implication:** Larger batch sizes $N$ allow capturing more mutual information — this is why CLIP trains with $N = 32{,}768$.

| Batch Size $N$ | $\log N$ (nats) | Max MI Captured |
|---------------|---------------|-----------------|
| 256 | 5.55 | 5.55 nats |
| 4,096 | 8.32 | 8.32 nats |
| 32,768 | 10.40 | 10.40 nats |
| 65,536 | 11.09 | 11.09 nats |

### 3. Gradient Analysis

The gradient of the per-sample loss with respect to similarity $S_{ij}$:

$$
\frac{\partial \ell_i}{\partial S_{ij}} = \frac{1}{\tau}\left(P_{ij} - \mathbb{1}[i=j]\right)
$$

where $P_{ij} = \text{softmax}(S_i / \tau)_j$ is the predicted probability.

**Gradient interpretation:**

| Pair type | $\mathbb{1}[i=j]$ | $P_{ij}$ | Gradient | Effect |
|-----------|--------|---------|----------|--------|
| Positive ($i=j$), well-aligned | 1 | $\approx 1$ | $\approx 0$ | No update needed |
| Positive ($i=j$), misaligned | 1 | $\ll 1$ | $\ll 0$ (negative) | Push $S_{ii}$ **up** |
| Hard negative ($i \neq j$, high $P$) | 0 | high | $> 0$ (positive) | Push $S_{ij}$ **down** strongly |
| Easy negative ($i \neq j$, low $P$) | 0 | $\approx 0$ | $\approx 0$ | Ignore (already separated) |

### 4. Temperature $\tau$ — Deep Analysis

$$
P(j \mid i) = \frac{e^{S_{ij}/\tau}}{\sum_k e^{S_{ik}/\tau}}
$$

**Effect on softmax variance:**

When similarities are i.i.d. with variance $\sigma_S^2$, the variance of the exponentiated scores:

$$
\text{Var}(e^{S/\tau}) \propto e^{\sigma_S^2/\tau^2}
$$

| $\tau$ | Distribution | Gradient Magnitude | Training Effect |
|--------|-------------|-------------------|----------------|
| Very small ($0.01$) | Near one-hot | Very large for non-top | Unstable, can diverge |
| CLIP default ($0.07$) | Peaked | Balanced | Good convergence |
| Standard ($1.0$) | Soft | Small, uniform | Slow learning |
| Large ($10.0$) | Near-uniform | Very small | Almost no learning |

**Learnable temperature:** CLIP makes $\tau$ learnable (actually $\log \tau$) and clips it to $[\tau_{\min}, \tau_{\max}]$:

$$
\tau = \exp(t), \quad t \leftarrow t - \eta \frac{\partial \mathcal{L}}{\partial t}, \quad \tau \in [0.01, 100]
$$

In practice, CLIP converges to $\tau \approx 0.01$ after training.

### 5. SigLIP — Sigmoid Loss (Softmax-Free Alternative)

Instead of softmax over the full $N \times N$ matrix, SigLIP uses **pairwise sigmoid** loss:

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N}\sum_{j=1}^{N} \left[ y_{ij} \log \sigma(S_{ij}) + (1-y_{ij}) \log(1 - \sigma(S_{ij})) \right]
$$

where $y_{ij} = \mathbb{1}[i = j]$ and $\sigma$ is the sigmoid function.

**SigLIP vs InfoNCE Comparison:**

| Property | InfoNCE | SigLIP |
|----------|---------|--------|
| Normalization | Softmax (global) | Sigmoid (pairwise) |
| Batch dependency | Yes (all-pairs softmax) | No (each pair independent) |
| Distributed training | Requires all-gather | Embarrassingly parallel |
| Memory | $O(N^2)$ softmax | $O(N^2)$ but simpler |
| Performance | Slightly better at small scale | Better at very large scale |

---

## Hard Negatives — Why Batch Size Matters

### Probability of a Hard Negative

A "hard negative" is a non-matching pair $(i, j)$ with high similarity. The probability of finding at least one hard negative in a batch of size $N$:

$$
P(\text{at least one hard neg}) = 1 - (1 - p_{\text{hard}})^{N-1}
$$

where $p_{\text{hard}}$ is the probability a random sample is a hard negative.

**Numerical example** (assume $p_{\text{hard}} = 0.01$):

| Batch Size $N$ | $P(\geq 1 \text{ hard neg})$ |
|---------------|----------------------------|
| 64 | 47% |
| 256 | 92% |
| 1,024 | 99.997% |
| 32,768 | $\approx 100$% |

This is why CLIP uses $N = 32{,}768$ — every batch is guaranteed to contain challenging negatives that force the model to learn fine-grained distinctions.

### Hard Negative Mining Strategies

```
  ┌─────────────────────────────────────────────────────┐
  │  Approach 1: Bigger Batches                          │
  │  Simple but memory-intensive                         │
  │  CLIP: N=32768, SigLIP: N=32768                     │
  │                                                      │
  │  Approach 2: Memory Bank                             │
  │  Store embeddings from past batches                  │
  │  MoCo-style: queue of K=65536 negatives             │
  │                                                      │
  │  Approach 3: In-Batch Hard Negative Mining           │
  │  Up-weight the hardest negatives:                    │
  │  w_ij = (S_ij)^β / Σ_k (S_ik)^β                    │
  │                                                      │
  │  Approach 4: Cross-GPU Gathering                     │
  │  AllGather embeddings across GPUs                    │
  │  8 GPUs × 4096 = effective N = 32768                │
  └─────────────────────────────────────────────────────┘
```

---

## Training Dynamics — What You'll See

```
  Similarity Matrix Evolution:

  Epoch 0 (random init)     Epoch 50 (learning)      Epoch 200 (converged)
  ┌───────────────┐         ┌───────────────┐        ┌───────────────┐
  │ .31 .38 .22   │         │ .65 .18 .12   │        │ .92 .03 .02   │
  │ .24 .29 .41   │   ──►   │ .11 .72 .09   │  ──►   │ .02 .94 .02   │
  │ .39 .21 .33   │         │ .15 .08 .68   │        │ .02 .02 .93   │
  └───────────────┘         └───────────────┘        └───────────────┘
   All entries ≈ 1/N         Diagonal emerging        Sharp identity-like

  Loss curve:
  4.0 ┤████
  3.0 ┤   ████
  2.0 ┤       █████
  1.0 ┤            ████████
  0.1 ┤                    ████████████████████
      └────┴────┴────┴────┴────┴────┴────┴────
       0   25   50   75  100  125  150  175  200
```

### What Each Phase Teaches

| Phase | Epochs | What the Model Learns |
|-------|--------|----------------------|
| Early (0-20) | Rapid loss drop | Coarse modality alignment (e.g., animals vs. vehicles) |
| Middle (20-100) | Slow, steady | Fine-grained distinctions (e.g., dog vs. cat) |
| Late (100+) | Plateau | Hard negatives (e.g., "golden retriever" vs. "labrador") |

---

## What You'll Build

- Full InfoNCE training loop from scratch
- Temperature sweep experiment ($\tau \in [0.01, 1.0]$)
- SigLIP implementation for comparison
- Similarity matrix visualization at every epoch
- Hard negative analysis: which pairs are hardest to distinguish

---

## Prerequisites

- Module 01 notebooks (cosine similarity, encoders)
- Module 02 Notebook 01 (CLIP architecture)
- Comfortable with softmax, cross-entropy, gradient computation

---

## Next Step

**[02_pretraining_objectives](../02_pretraining_objectives/)** — Beyond contrastive: ITC, ITM, MLM, Generation
