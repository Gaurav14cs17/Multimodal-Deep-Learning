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

### 2. InfoNCE Loss — Full Step-by-Step Derivation

**Step 1:** Start with mutual information lower bound (InfoMax):

$$
I(X; Y) \geq \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\log\frac{f(x_i, y_i)}{\frac{1}{N}\sum_{j=1}^{N}f(x_i, y_j)}\right]
$$

**Step 2:** Choose $f(x, y) = \exp(x^\top y / \tau)$ (unnormalized density ratio on the hypersphere):

$$
\mathcal{L}_{\text{NCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(x_i^\top y_i / \tau)}{\sum_{j=1}^{N}\exp(x_i^\top y_j / \tau)}
$$

**Step 3:** Compute gradient w.r.t. temperature:

$$
\frac{\partial \mathcal{L}}{\partial \tau} = \frac{1}{\tau^2}\sum_{i}\sum_{j}p_{ij}s_{ij} - \frac{1}{\tau^2}\sum_{i}s_{ii}
$$

where $p_{ij} = \text{softmax}(s_{ij}/\tau)$ and $s_{ij} = x_i^\top y_j$.

**Numerical Example:** Given 3 samples with similarity scores:

- $s_{11} = 0.9, s_{12} = 0.2, s_{13} = 0.1$
- $\tau = 0.07$

$$
p_{11} = \frac{e^{0.9/0.07}}{e^{0.9/0.07} + e^{0.2/0.07} + e^{0.1/0.07}} = \frac{e^{12.86}}{e^{12.86} + e^{2.86} + e^{1.43}} = 0.9999
$$

$$
\mathcal{L}_1 = -\log(0.9999) \approx 0.0001
$$

Symmetric CLIP loss averages image→text and text→image directions.

### 3. Mutual Information Lower Bound

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

### 4. Gradient Analysis

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

### 5. Temperature $\tau$ — Deep Analysis

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

### 6. SigLIP — Sigmoid Loss (Softmax-Free Alternative)

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

## Mathematical Proofs

### Proof: InfoNCE Gradient $\partial \mathcal{L}/\partial f(x_i, y_i)$

**Setup:** $f(x_i, y_j) = \text{sim}(x_i, y_j)/\tau$, $\mathcal{L}_i = -\log \frac{e^{f(x_i,y_i)}}{\sum_j e^{f(x_i,y_j)}}$.

**Step 1 — Softmax probabilities:**

$$
p_{ij} = \frac{e^{f(x_i, y_j)}}{\sum_k e^{f(x_i, y_k)}}
$$

**Step 2 — Derivative of log-softmax:**

$$
\frac{\partial \mathcal{L}_i}{\partial f(x_i, y_j)} = p_{ij} - \mathbb{1}[j = i]
$$

**Step 3 — Interpretation:**

- Positive pair ($j=i$): gradient $= p_{ii} - 1 \leq 0$ → increase $f(x_i, y_i)$
- Negative pair ($j \neq i$): gradient $= p_{ij} > 0$ → decrease $f(x_i, y_j)$

**Step 4 — Chain rule to embeddings:** $\partial f / \partial x_i = (y_j - \sum_k p_{ik} y_k)/\tau$ for contrastive learning on the sphere. **∎**

#### Numerical Example

$f = [2.0, 0.5, 0.1]$ (positive at index 0): $p \approx [0.78, 0.14, 0.08]$. $\partial \mathcal{L}/\partial f_0 = 0.78 - 1 = -0.22$ (push $f_0$ up); $\partial \mathcal{L}/\partial f_1 = 0.14$ (push $f_1$ down).

---

### Proof: Low $\tau$ Makes Hard Negatives Dominate Gradients

**Claim:** As $\tau \to 0$, softmax weights concentrate on the hardest negative (highest similarity among $j \neq i$).

**Step 1 — Softmax with temperature:**

$$
p_{ij} = \frac{e^{S_{ij}/\tau}}{\sum_k e^{S_{ik}/\tau}
$$

**Step 2 — Limit analysis:** Let $S_{i,i^*}$ be the max over negatives $j \neq i$. As $\tau \to 0$:

$$
p_{ij} \to \begin{cases} 1 & j = i \text{ if } S_{ii} > S_{i,i^*} \\ 0 & \text{otherwise} \end{cases}, \quad p_{i,i^*} \to 1 \text{ among negatives if } S_{ii} \leq S_{i,i^*}
$$

**Step 3 — Gradient magnitude on hard negative:**

$$
\frac{\partial \mathcal{L}}{\partial S_{i,i^*}} = \frac{1}{\tau} p_{i,i^*}
$$

For small $\tau$, $p_{i,i^*}$ remains significant only for near-tie negatives — gradient scales as $1/\tau$, amplifying hard-negative signal. **∎**

#### Numerical Example

$S_i = [0.9, 0.85, 0.1]$, $\tau = 0.07$: $p_i \approx [0.52, 0.47, 0.01]$. Hard negative (index 1) gets 47% of gradient mass despite being a negative. At $\tau = 1.0$: $p_i \approx [0.38, 0.36, 0.26]$ — gradients spread uniformly.

---

### Proof: SigLIP from Binary Cross-Entropy

**Step 1 — Pairwise labels:** $y_{ij} = \mathbb{1}[i = j]$ (positive if matched, else negative).

**Step 2 — Independent sigmoid for each pair:**

$$
P(y_{ij} = 1) = \sigma(S_{ij}) = \frac{1}{1 + e^{-S_{ij}}}
$$

**Step 3 — BCE loss per pair:**

$$
\ell_{ij} = -y_{ij} \log \sigma(S_{ij}) - (1 - y_{ij}) \log(1 - \sigma(S_{ij}))
$$

**Step 4 — Average over all $N^2$ pairs:**

$$
\mathcal{L}_{\text{SigLIP}} = \frac{1}{N^2} \sum_{i,j} \ell_{ij}
$$

**Why vs InfoNCE:** No softmax normalization across batch — each pair is independent, enabling distributed training without all-gather. **∎**

#### Numerical Example

$S_{11} = 2.0$ (positive), $S_{12} = 0.5$ (negative): $\ell_{11} = -\log(0.88) = 0.128$; $\ell_{12} = -\log(0.38) = 0.968$. Total contributes $1.096$ for this row pair.

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

## 🔬 Worked Examples in the Notebook

### SigLIP vs InfoNCE — Side-by-Side Comparison
- Implement both loss functions from scratch
- Train on same data for 200 steps
- Compare convergence curves
- Comparison table: normalization, distributed scaling, memory, used-in models

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning/01_contrastive_learning.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| SimCLR Framework (`../../assets/paper_figures/simclr_framework.png`) | Chen et al. — [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) | 2020 | Augmentation → encoder → projection → NT-Xent loss |
| MoCo Memory Bank | He et al. — [arXiv:1911.05722](https://arxiv.org/abs/1911.05722) | 2020 | Momentum encoder + queue of 65K negatives |

### Multimodal RLHF

Beyond text-only RLHF, vision-language models need alignment too:

| Method | Paper | Key Innovation |
|--------|-------|----------------|
| **RLHF-V** | Yu et al. (2024) — [arXiv:2312.00849](https://arxiv.org/abs/2312.00849) | Fine-grained segment-level human feedback |
| **LLaVA-RLHF** | Sun et al. (2023) — [arXiv:2309.14525](https://arxiv.org/abs/2309.14525) | First systematic RLHF study for MLLMs |
| **Silkie** | Li et al. (2023) | GPT-4V AI feedback + DPO training |

### Additional Papers Covered

- **DINO** (Caron et al., 2021) — Self-distillation with no labels, EMA teacher
- **DINOv2** (Oquab et al., 2023) — Universal visual features, iBOT objective
- **SigLIP** (Zhai et al., 2023) — Sigmoid pairwise contrastive loss

---

## Next Step

**[02_pretraining_objectives](../02_pretraining_objectives/)** — Beyond contrastive: ITC, ITM, MLM, Generation
