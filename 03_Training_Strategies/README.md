# Module 03: Training Strategies

> **Time:** 2–3 hours | **Notebooks:** 5 | **Visuals:** 11+ plots | **Difficulty:** Intermediate–Advanced
>
> **This is a CORE FOCUS module** — mastering training is what separates "I understand the theory" from "I can actually build this."

| Notebook | Open in Colab |
|----------|---------------|
| `01_contrastive_learning/01_contrastive_learning.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning/01_contrastive_learning.ipynb) |
| `02_pretraining_objectives/02_pretraining_objectives.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives/02_pretraining_objectives.ipynb) |
| `03_training_pipeline/03_training_pipeline.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline/03_training_pipeline.ipynb) |
| `04_multimodal_alignment/04_multimodal_alignment.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/04_multimodal_alignment/04_multimodal_alignment.ipynb) |
| `05_scaling_laws/05_scaling_laws.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/05_scaling_laws/05_scaling_laws.ipynb) |

---

## Why This Module Matters

> Building the architecture is **20% of the work**. Training it correctly is **80%**.

```
  Architecture                Training
  ============                ========

  Module 01–02                Module 03 (THIS MODULE)
  "How to build"              "How to make it work"

  +--------+                  +--------+--------+--------+
  | Design |                  |  Loss  | Optim  | Scale  |
  | model  |   -------->      |  func  | tricks | tricks |
  +--------+                  +--------+--------+--------+
                                 |         |         |
   20% of success               |    80% of success  |
                                 v         v         v
                              WORKING MODEL
```

This module treats multimodal training as an optimization problem over joint embedding geometry, multi-objective pretraining, and the numerical engineering that makes billion-parameter runs stable on finite hardware.

---

## Notebook 1: `01_contrastive_learning/01_contrastive_learning.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning/01_contrastive_learning.ipynb)

### 1. InfoNCE Loss — Full Symmetric Formula

Given a batch of $N$ image–text pairs, encode images and texts into unit-norm embeddings $\mathbf{v}_i, \mathbf{t}_j \in \mathbb{R}^d$ and form the cosine similarity matrix:

$$S_{ij} = \mathbf{v}_i^\top \mathbf{t}_j = \frac{\mathbf{v}_i^\top \mathbf{t}_j}{\|\mathbf{v}_i\| \|\mathbf{t}_j\|} \in [-1, 1]$$

InfoNCE treats each row and each column as an $N$-way softmax classification problem. The **symmetric** CLIP loss is:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \underbrace{\log \frac{e^{S_{ii}/\tau}}{\sum_{j=1}^{N} e^{S_{ij}/\tau}}}_{\text{image } i \to \text{which text?}} + \underbrace{\log \frac{e^{S_{ii}/\tau}}{\sum_{j=1}^{N} e^{S_{ji}/\tau}}}_{\text{text } i \to \text{which image?}} \right]$$

Define row-softmax probabilities $P_{ij}^{\text{i→t}} = \dfrac{e^{S_{ij}/\tau}}{\sum_{k=1}^{N} e^{S_{ik}/\tau}}$ and column-softmax $P_{ij}^{\text{t→i}} = \dfrac{e^{S_{ij}/\tau}}{\sum_{k=1}^{N} e^{S_{kj}/\tau}}$. Then:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log P_{ii}^{\text{i→t}} + \log P_{ii}^{\text{t→i}} \right]$$

Each term is a cross-entropy over $N$ classes where the correct class is the diagonal index $i$.

---

### 2. Connection to Mutual Information

InfoNCE is not merely a heuristic — it lower-bounds mutual information. Let $V$ denote the image representation and $T$ the text representation for a positive pair drawn from the joint distribution $p(v, t)$. With $N$ samples where one is positive and $N-1$ are negatives drawn from the marginal $p(v)p(t)$, Oord et al. show:

$$\mathcal{L}_{\text{InfoNCE}} \;\geq\; \log N - I(V; T)$$

Rearranging:

$$I(V; T) \;\geq\; \log N - \mathcal{L}_{\text{InfoNCE}}$$

**Interpretation:** Minimizing InfoNCE **maximizes a lower bound** on $I(V;T)$. Larger batch size $N$ tightens the bound (larger $\log N$), which is one mathematical reason CLIP trains with $N = 32{,}768$. The bound becomes loose when negatives are easy (high $\mathcal{L}$) or when $N$ is small.

---

### 3. Full Step-by-Step Numerical Example ($N=4$, $\tau=0.07$)

**Given** similarity matrix $S$ (cosine similarities):

$$
S = \begin{pmatrix} 0.5 & 0.1 & 0.3 & 0.2 \\ 0.2 & 0.6 & 0.1 & 0.3 \\ 0.3 & 0.2 & 0.4 & 0.1 \\ 0.1 & 0.3 & 0.2 & 0.5 \end{pmatrix}
$$

**Step 1 — Scale by temperature** $\tau = 0.07$:

$$
\frac{S}{\tau} = \begin{pmatrix} 7.143 & 1.429 & 4.286 & 2.857 \\ 2.857 & 8.571 & 1.429 & 4.286 \\ 4.286 & 2.857 & 5.714 & 1.429 \\ 1.429 & 4.286 & 2.857 & 7.143 \end{pmatrix}
$$

**Step 2 — Row-wise softmax** $P_{ij}^{\text{i→t}} = \dfrac{e^{S_{ij}/\tau}}{\sum_k e^{S_{ik}/\tau}}$:

| Row $i$ | $e^{S_{i1}/\tau}$ | $e^{S_{i2}/\tau}$ | $e^{S_{i3}/\tau}$ | $e^{S_{i4}/\tau}$ | Row sum | $P_{ii}$ |
|---------|-------------------|-------------------|-------------------|-------------------|---------|----------|
| 1 | 1265.04 | 4.173 | 72.654 | 17.412 | 1359.28 | **0.9307** |
| 2 | 17.412 | 5278.67 | 4.173 | 72.654 | 5372.90 | **0.9825** |
| 3 | 72.654 | 17.412 | 303.17 | 4.173 | 397.41 | **0.7629** |
| 4 | 4.173 | 72.654 | 17.412 | 1265.04 | 1359.28 | **0.9307** |

Detailed row 1: $e^{7.143} = 1265.04$, $e^{1.429} = 4.173$, $e^{4.286} = 72.654$, $e^{2.857} = 17.412$. Sum $= 1359.28$. So $P_{11} = 1265.04 / 1359.28 = 0.9307$.

Full row-1 probability vector: $P_{1,:} = [0.9307,\; 0.0031,\; 0.0534,\; 0.0128]$.

**Step 3 — Column-wise softmax** (text → image), $P_{ij}^{\text{t→i}} = \dfrac{e^{S_{ij}/\tau}}{\sum_k e^{S_{kj}/\tau}}$:

| Col $j$ | Diagonal $P_{jj}^{\text{t→i}}$ |
|---------|-------------------------------|
| 1 | $1265.04 / 1359.28 = \mathbf{0.9307}$ |
| 2 | $5278.67 / 5372.90 = \mathbf{0.9825}$ |
| 3 | $303.17 / 397.41 = \mathbf{0.7629}$ |
| 4 | $1265.04 / 1359.28 = \mathbf{0.9307}$ |

**Step 4 — Negative log of diagonal, averaged:**

Image→text term:
$$-\frac{1}{4}\sum_i \log P_{ii}^{\text{i→t}} = -\frac{1}{4}\bigl[\log(0.9307) + \log(0.9825) + \log(0.7629) + \log(0.9307)\bigr] = 0.1080$$

Text→image term (identical here): $0.1080$

**Step 5 — Symmetric InfoNCE:**

$$\mathcal{L}_{\text{InfoNCE}} = \frac{0.1080 + 0.1080}{2} = \boxed{0.1080}$$

The loss is modest because diagonals already dominate each softmax, but off-diagonal mass (e.g. $P_{13} = 72.654/1359.28 = 0.0534$) still pulls gradients toward hard negatives.

---

### 4. Temperature Analysis

#### 4a. Variance of Dot Products

Let $\mathbf{q}, \mathbf{k} \in \mathbb{R}^d$ have i.i.d. entries $q_\ell, k_\ell \sim \mathcal{N}(0, 1)$. The dot product is:

$$\mathbf{q}^\top \mathbf{k} = \sum_{\ell=1}^{d} q_\ell k_\ell$$

Each term has $\mathbb{E}[q_\ell k_\ell] = 0$ and, by independence,

$$\mathbb{E}[q_\ell^2 k_\ell^2] = \mathbb{E}[q_\ell^2]\,\mathbb{E}[k_\ell^2] = 1 \cdot 1 = 1 = \mathrm{Var}(q_\ell k_\ell)$$

Summing $d$ independent zero-mean terms:

$$\boxed{\mathrm{Var}(\mathbf{q}^\top \mathbf{k}) = d}$$

For $d = 512$, typical dot-product std is $\sqrt{512} \approx 22.6$ before normalization. L2-normalization maps similarities into $[-1,1]$, but raw logits still scale with dimension — temperature $\tau$ re-scales them for stable softmax dynamics.

#### 4b. Softmax Sharpness and Gradients

$$P_j = \frac{e^{s_j / \tau}}{\sum_k e^{s_k / \tau}}$$

As $\tau \to 0^+$: $P_j \to \mathbb{1}[j = \arg\max_k s_k]$ (hard winner-take-all). As $\tau \to \infty$: $P_j \to 1/N$ (uniform, no learning signal).

For a single row with loss $\ell = -\log P_i$ where $i$ is the positive index:

$$\frac{\partial \ell}{\partial S_{ij}} = \frac{1}{\tau}\left(P_{ij} - \mathbb{1}[i = j]\right)$$

| Pair type | Gradient sign | Effect |
|-----------|---------------|--------|
| Positive ($i = j$) | $\frac{1}{\tau}(P_{ii} - 1) < 0$ | Increase $S_{ii}$ |
| Negative ($i \neq j$) | $\frac{1}{\tau} P_{ij} > 0$ | Decrease $S_{ij}$ |

**Hard negatives** ($P_{ij}$ large) receive **larger** gradients — the loss automatically focuses on confusing pairs. Small $\tau$ amplifies all gradients by $1/\tau$ but also sharpens $P$, concentrating mass on the hardest negative.

#### 4c. Learnable Temperature: $\tau = e^t$

CLIP parameterizes $\tau = e^t$ with learnable scalar $t$, often clamped to $[0, \log 100]$ so $\tau \in [1, 100]$. Benefits:

- $\tau > 0$ always (log-parameterization)
- Multiplicative updates in $\tau$ become additive in $t$
- The model can learn whether the task needs sharp ($\tau \downarrow$) or soft ($\tau \uparrow$) comparisons

---

### 5. SigLIP — A Softmax-Free Alternative

SigLIP replaces row-wise softmax with a sigmoid on **every** pair $(i,j)$:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma\!\left( z_{ij} \cdot \frac{s_{ij}}{\tau} \right), \quad z_{ij} = 2\,\mathbb{1}[i = j] - 1$$

So positives ($i = j$) use $\sigma(+s/\tau)$ and negatives ($i \neq j$) use $\sigma(-s/\tau)$. There is **no softmax partition function** — each pair is an independent binary logistic problem.

| Property | InfoNCE (CLIP) | SigLIP |
|----------|----------------|--------|
| Normalization | Row softmax ($O(N)$ per row) | None ($O(1)$ per pair) |
| Batch scaling | Softmax saturates at large $N$ | Scales naturally to large batches |
| Loss coupling | Pairs compete within row | Pairs decoupled |

SigLIP's pairwise structure avoids the $\log N$ softmax denominator bottleneck and is the loss of choice in many large-scale vision–language trainers (e.g. SigLIP, some PaLI variants).

---

### 6. Hard Negatives — Probability Analysis

In batch size $N$, each positive pair faces $N - 1$ in-batch negatives. Let $p_h$ be the probability that **one random negative** is "hard" (high similarity, semantically confusable). Assuming independence:

$$P(\text{at least one hard negative}) = 1 - (1 - p_h)^{N-1}$$

For small $p_h$: $1 - (1-p_h)^{N-1} \approx 1 - e^{-(N-1)p_h}$.

| $N$ | $p_h = 0.01$ | $p_h = 0.05$ | $p_h = 0.10$ |
|-----|--------------|--------------|--------------|
| 32 | 0.275 | 0.798 | 0.968 |
| 256 | 0.920 | ≈ 1.000 | ≈ 1.000 |
| 32,768 | ≈ 1.000 | ≈ 1.000 | ≈ 1.000 |

CLIP's $N = 32{,}768$ virtually guarantees hard negatives every step — a key ingredient for fine-grained alignment. On limited GPU memory, **gradient accumulation** (Notebook 3) simulates large $N$ without storing all activations at once.

---

### 7. Training Evolution — Similarity Matrix Snapshots

Six epochs of an idealized $3 \times 3$ similarity matrix (values rounded):

```
Epoch 0 (random)          Epoch 5 (starting)        Epoch 20 (emerging)
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ .31  .38  .22   │       │ .28  .35  .24   │       │ .41  .19  .12   │
│ .24  .29  .41   │       │ .22  .44  .31   │       │ .18  .52  .21   │
│ .39  .21  .33   │       │ .30  .23  .42   │       │ .11  .20  .55   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
 off-diagonal ≈ diagonal    diagonal creeping up      structure visible

Epoch 50 (forming)        Epoch 100 (strong)         Epoch 200 (converged)
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ .62  .11  .09   │       │ .81  .06  .04   │       │ .92  .03  .02   │
│ .10  .71  .08   │       │ .05  .89  .05   │       │ .02  .94  .02   │
│ .08  .09  .68   │       │ .04  .04  .87   │       │ .02  .02  .93   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
 diagonal dominant          near block-diagonal         ≈ identity structure
```

The diagonal emerges because positives are pushed toward $S_{ii} \to 1$ while negatives are pushed toward $S_{ij} \to -1$ (or near zero before normalization).

---

### 8. Embedding Space — Before vs After Training

**Before training** (random init — isotropic cloud, pairs unrelated):

```
        t
        ^
   ·  ·   ·  ·
  ·   · ·    ·        v = image embeddings (circles)
   · ·   · ·          t = text embeddings (x)
  ·  x   ·  ·
 ·   · x  ·   ·
 ·  ·   ·  x
  · x  ·   ·
   ·  · ·  ·
        +----------> v

  No alignment: nearest neighbor rarely the paired modality.
```

**After contrastive training** (paired clusters):

```
        t
        ^
              x₁ ←──→ ·₁     Pair 1: image-text collapsed
         x₂ ←──→ ·₂          Pair 2
    x₃ ←──→ ·₃               Pair 3
              ·₄ ←──→ x₄     Pair 4

  ·ᵢ and xᵢ co-locate; other pairs repelled into separate directions.
  Cosine similarity matrix ≈ I (identity) at convergence.
```

Geometrically, InfoNCE performs **simultaneous attraction** (positive pairs) and **repulsion** (negatives) on the unit hypersphere $\mathbb{S}^{d-1}$.

---

## Notebook 2: `02_pretraining_objectives/02_pretraining_objectives.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives/02_pretraining_objectives.ipynb)

### 1. Multi-Objective Loss

Modern vision–language pretraining (BLIP, BLIP-2, CoCa) combines complementary objectives:

$$\mathcal{L} = \alpha\,\mathcal{L}_{\text{ITC}} + \beta\,\mathcal{L}_{\text{ITM}} + \gamma\,\mathcal{L}_{\text{MLM}} + \delta\,\mathcal{L}_{\text{Gen}}$$

Each term supervises a different granularity: global alignment (ITC), fine-grained fusion (ITM), token-level grounding (MLM), and open-ended generation (Gen). Weights $\alpha, \beta, \gamma, \delta$ are typically chosen so gradient magnitudes are comparable (often $\alpha = \beta = 1$, $\gamma \in [0.5, 1]$).

---

### 2. ITC — Image-Text Contrastive

Identical in spirit to CLIP InfoNCE (often image→text only, or symmetric):

$$\mathcal{L}_{\text{ITC}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N} e^{s_{ij}/\tau}}, \quad s_{ij} = \mathbf{v}_i^\top \mathbf{t}_j$$

Provides **modality-level** alignment without fusion — fast, scalable, but blind to token-level detail.

---

### 3. ITM — Image-Text Matching with Hard Negative Mining

ITM is binary cross-entropy on **fused** cross-modal representations. A multimodal encoder produces $\mathbf{h}_i^{\text{fused}} = f_{\text{fuse}}(\mathbf{v}_i, \mathbf{t}_i)$ and a linear head outputs logit $z_i = \mathbf{w}^\top \mathbf{h}_i^{\text{fused}}$:

$$\mathcal{L}_{\text{ITM}} = -\sum_{i \in \mathcal{B}} \Bigl[ y_i \log \sigma(z_i) + (1 - y_i) \log\bigl(1 - \sigma(z_i)\bigr) \Bigr]$$

where $y_i = 1$ for true pairs and $y_i = 0$ for negatives.

**Hard negative mining via ITC:** For each image $i$, sort $\{s_{ij}\}_{j \neq i}$ descending and select the text $j^* = \arg\max_{j \neq i} s_{ij}$ as the hard negative. Symmetrically, for each text, select the hardest image. Only hard negatives are fed to the fusion encoder — easy negatives waste compute and provide weak gradients.

```
  Image i ──ITC scores──> [0.12, 0.08, 0.71, 0.05]  ──pick j*=3 (hardest)
                              │
                              v
                    Fuse(v_i, t_3) ──> ITM head ──> y=0 (negative)
```

---

### 4. MLM — Masked Language Modeling with Image Context

Mask 15% of text tokens (80% `[MASK]`, 10% random, 10% unchanged — BERT protocol). Let $\mathcal{M}$ be masked positions, $\mathbf{w}_{\backslash \mathcal{M}}$ unmasked tokens, $I$ the image:

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P\!\left(w_t \,\middle|\, \mathbf{w}_{\backslash \mathcal{M}},\, I\right)$$

The cross-modal encoder attends over image patches **and** text tokens, so predicting $w_t$ requires visual grounding (e.g. "The `[MASK]` is red" + image of a car → "car"). MLM teaches **fine-grained** correspondence that ITC alone cannot.

---

### 5. Generation — Autoregressive Language Modeling

Caption generation with causal masking:

$$\mathcal{L}_{\text{Gen}} = -\sum_{t=1}^{T} \log P(w_t \mid w_1, \ldots, w_{t-1},\, I)$$

Equivalently, minimize cross-entropy over the vocabulary at each timestep with image cross-attention (or prefix tokens). This objective enables **downstream generation** (captioning, VQA decoding) without task-specific heads.

---

### 6. BLIP Architecture (ASCII)

```
                    ┌──────────────────────────────────────────────────────┐
                    │                    BLIP (Li et al.)                   │
                    └──────────────────────────────────────────────────────┘

  Image ──> [ViT Encoder] ──> v_i ──┬──> ITC head (dual encoders, shared weights
                                    │     between ITC image/text towers)
                                    │
                                    ├──> [Cross-Modal Encoder] ──> h_fused ──> ITM head
                                    │         ^                           (binary match)
                                    │         │
  Text  ──> [BERT Encoder] ──> t_i ──┘         │
           (shared for ITC                     │
            text tower)                        │
                                               ├──> MLM head (predict masked tokens)
                                               │
                                               └──> [Decoder] ──> Gen (autoregressive)

  SHARED: Image ViT (ITC + fusion input), Text BERT (ITC + fusion input)
  UNSHARED: ITC projection heads, ITM classifier, MLM head, causal decoder
```

The **Mediation trick:** BLIP uses the same encoders for contrastive (shallow, dual) and fusion (deep, cross-attention) paths, amortizing representation learning across four objectives.

---

### 7. Budget → Objectives → Quality → Model

| Budget | Objectives | Alignment Quality | Representative Model |
|--------|------------|-------------------|----------------------|
| Very Low | ITC only | Global retrieval; weak grounding | CLIP, SigLIP |
| Low | ITC + ITM | + fine-grained match discrimination | ALBEF |
| Medium | ITC + ITM + MLM | + token-level visual grounding | BLIP |
| High | ITC + ITM + MLM + Gen | + generative fluency & transfer | BLIP-2, CoCa |

**Rule of thumb:** Each added objective increases compute ~1.3–2× but closes a specific capability gap. ITC alone never teaches "which word aligns to which region."

---

## Notebook 3: `03_training_pipeline/03_training_pipeline.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline/03_training_pipeline.ipynb)

### 1. Optimizer Evolution: SGD → Momentum → Adam → AdamW

#### SGD

$$\theta_{t+1} = \theta_t - \eta\, g_t, \quad g_t = \nabla_\theta \mathcal{L}(\theta_t)$$

Simple but noisy; learning rate must compensate for gradient variance.

#### Momentum

Accumulate an exponential moving average of gradients:

$$m_t = \beta\, m_{t-1} + g_t, \quad \theta_{t+1} = \theta_t - \eta\, m_t$$

Typical $\beta = 0.9$. Momentum smooths oscillations and accelerates in consistent directions (physical analogy: velocity).

#### Adam — Adaptive Moments with Bias Correction

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \quad \text{(first moment)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \quad \text{(second moment)}$$

Raw updates use $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Why bias correction?** At $t = 1$ with $m_0 = 0$:

$$m_1 = (1 - \beta_1)\, g_1$$

But $\mathbb{E}[m_t] = \mathbb{E}[g_t]$ at stationarity requires $m_t$ to estimate the mean gradient, not $(1-\beta_1^t)$ times it. Since $\mathbb{E}[m_1] = (1-\beta_1)\mathbb{E}[g_1]$, dividing by $(1 - \beta_1^1) = (1 - \beta_1)$ recovers an unbiased estimate:

$$\mathbb{E}\!\left[\frac{m_1}{1 - \beta_1}\right] = \mathbb{E}[g_1]$$

Similarly for $v_t$ with $(1 - \beta_2^t)$. Without correction, early steps use effective learning rates scaled by $(1 - \beta_1^t) \ll 1$ — too conservative.

#### AdamW — Decoupled Weight Decay

Standard Adam with L2 regularization adds $\lambda \theta$ **into the gradient**:

$$g_t \leftarrow g_t + \lambda \theta_t \quad \Rightarrow \quad \text{weight decay is scaled by } 1/(\sqrt{\hat{v}_t} + \epsilon)$$

Adaptive per-parameter scaling **distorts** L2 regularization — some weights decay faster than intended.

**AdamW** decouples decay from the adaptive step:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Weight decay applies **uniformly** in parameter space, independent of gradient history. This is the default for transformer and CLIP-style training ($\lambda \approx 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).

---

### 2. Cosine Learning Rate with Warmup — Complete Formula

Let $t$ be the global step, $T_w = T_{\text{warmup}}$, $T_{\text{tot}} = T_{\text{total}}$, $\eta_{\max}$ peak LR, $\eta_{\min}$ floor LR:

$$
\eta(t) = \begin{cases}
\displaystyle \eta_{\max} \cdot \frac{t}{T_w} & \text{if } t < T_w \\[10pt]
\displaystyle \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\pi \cdot \frac{t - T_w}{T_{\text{tot}} - T_w}\right)\right) & \text{if } T_w \leq t \leq T_{\text{tot}}
\end{cases}
$$

Warmup prevents large Adam updates when $m_t, v_t$ are poorly estimated. Cosine decay provides smooth annealing without discrete step jumps.

```
   η(t)
   ^
ηmax|          *****
   |        **     **
   |       *         ***
   |      *              ****
   |     *                    *****
   |    *                          ******
   |   *  ← linear warmup
   |  *
ηmin| *________________________________________
   +----+--------------------------------------> t
   0   Tw                                  Ttot
```

---

### 3. Gradient Accumulation

#### Mathematical Equivalence

Process $K$ micro-batches $\mathcal{B}_1, \ldots, \mathcal{B}_K$ before one optimizer step. Accumulate:

$$\bar{g} = \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta \mathcal{L}(\mathcal{B}_k)$$

**Claim:** $\bar{g} \approx \nabla_\theta \mathcal{L}\!\left(\bigcup_{k=1}^{K} \mathcal{B}_k\right)$ when $\mathcal{L}$ is averaged over samples.

**Proof sketch:** If $\mathcal{L}(\mathcal{B}) = \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}} \ell(x)$ and micro-batches are disjoint with $|\mathcal{B}_k| = B$:

$$\nabla_\theta \mathcal{L}\!\left(\bigcup_k \mathcal{B}_k\right) = \frac{1}{KB}\sum_{k=1}^{K}\sum_{x \in \mathcal{B}_k} \nabla_\theta \ell(x) = \frac{1}{K}\sum_{k=1}^{K} \nabla_\theta \mathcal{L}(\mathcal{B}_k) = \bar{g}$$

Exact equality holds for **loss = mean** over micro-batch samples; using **sum** loss requires dividing accumulated gradients by $K$.

#### Code Pattern

```python
optimizer.zero_grad()
for k, micro_batch in enumerate(dataloader):
    loss = model(micro_batch) / K          # scale loss so mean over K steps = full-batch mean
    loss.backward()                         # gradients accumulate in .grad
    if (k + 1) % K == 0:
        clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

#### Memory vs Effective Batch Size

| Physical batch $B$ | Accumulation $K$ | Effective batch $B_{\text{eff}} = BK$ | Activation memory | Optimizer memory |
|--------------------|------------------|---------------------------------------|-------------------|------------------|
| 32 | 1 | 32 | $O(B)$ | $O(\|\theta\|)$ |
| 32 | 4 | 128 | $O(B)$ — same | $O(\|\theta\|)$ — same |
| 32 | 32 | 1024 | $O(B)$ — same | $O(\|\theta\|)$ — same |
| 256 | 1 | 256 | $O(B)$ — 8× more | $O(\|\theta\|)$ |

Only **activations** scale with physical batch; accumulation trades **time** (more forward/backward passes) for **effective batch size**.

---

### 4. Gradient Clipping (Max Norm)

Let $g = \nabla_\theta \mathcal{L}$ be the full parameter gradient vector. Global norm clipping with threshold $c > 0$:

$$
\hat{g} = \begin{cases}
g & \text{if } \lVert g \rVert_2 \leq c \\
\displaystyle c \cdot \frac{g}{\lVert g \rVert_2} & \text{if } \lVert g \rVert_2 > c
\end{cases}
$$

**Direction preservation:** When clipping activates, $\hat{g} = c \cdot \hat{u}$ where $\hat{u} = g / \|g\|_2$ is the unit vector in direction $g$. Only magnitude changes.

**When it activates:** Early training, mixed precision without scaling, or very deep fusion encoders — whenever $\|g\|_2$ spikes. Typical $c = 1.0$ for transformers. If clipping fires every step, investigate loss scale or learning rate instead of raising $c$ indefinitely.

---

### 5. Mixed Precision

| Format | Exponent | Mantissa | Approx. range | Precision | Training notes |
|--------|----------|----------|---------------|-----------|----------------|
| fp32 | 8 bit | 23 bit | $\pm 3.4 \times 10^{38}$ | ~7 decimal digits | Master weights; reference |
| fp16 | 5 bit | 10 bit | $\pm 6.5 \times 10^{4}$ | ~3 decimal digits | Fast on Tensor Cores; needs loss scaling |
| bf16 | 8 bit | 7 bit | $\pm 3.4 \times 10^{38}$ | ~2 decimal digits | Same range as fp32; often no loss scaling |

**Loss scaling for fp16:** Gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ can underflow fp16's minimum normal ($\approx 6 \times 10^{-8}$). Multiply loss before backward:

$$\mathcal{L}_{\text{scaled}} = 2^{s} \cdot \mathcal{L}, \quad g_{\text{scaled}} = 2^{s} \cdot g$$

Then unscale: $g = g_{\text{scaled}} / 2^{s}$ before `optimizer.step()`. Dynamic scaling increases $s$ when overflow is absent, decreases on Inf/NaN.

**Typical pattern:** fp32 master weights + fp16/bf16 forward/backward → ~50% memory reduction, ~1.5–2× throughput on modern GPUs.

---

### 6. Gradient Checkpointing

Standard backprop stores **all** layer activations for the backward pass. For $L$ layers, activation memory is $O(L)$.

**Checkpointing:** Save activations only at $\sqrt{L}$ segment boundaries; recompute intermediate activations during backward by re-running forward locally.

| Strategy | Activation memory | Extra compute |
|----------|-------------------|---------------|
| No checkpointing | $O(L)$ | 0% |
| Segment checkpointing | $O(\sqrt{L})$ | ~33% (one extra forward per segment) |

For a 24-layer ViT, checkpointing can cut activation memory ~60%, enabling larger batch or sequence length at the cost of ~33% longer backward pass — almost always favorable when memory-bound.

```
  Forward (store checkpoints ● only):

  ●─── layer 1–4 ───●─── layer 5–8 ───●─── ... ───●
                    ↑ recompute 5–8 during backward

  Backward: load ●, re-run forward for segment, then backward through segment
```

---

## Real-World Multi-Stage Training Pipelines

Production multimodal models are never trained in a single stage. The modern recipe is a **3-stage pipeline** that progressively sharpens the model from raw pretraining to task-specific excellence. This pattern appears in LLaVA, InternVL, Qwen-VL, LightOnOCR, and virtually every state-of-the-art VLM.

### The Universal 3-Stage Pattern

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         THE 3-STAGE TRAINING PIPELINE                            │
│                                                                                  │
│                          ┌─────────────────────┐                                 │
│                          │   RAW BASE MODEL     │                                 │
│                          │ (random / pretrained) │                                 │
│                          └─────────┬───────────┘                                 │
│                                    │                                              │
│      ┌─────────────────────────────▼──────────────────────────────┐              │
│      │  STAGE 1: SUPERVISED FINE-TUNING (SFT)                     │              │
│      │  ─────────────────────────────────────                      │              │
│      │  • Next-token prediction on curated (image, text) pairs     │              │
│      │  • Optional: knowledge distillation from teacher model      │              │
│      │  • Data: 100K–50M instruction pairs                         │              │
│      │  • Learns: OCR, captioning, spatial reasoning, formatting   │              │
│      │  • Math: L = -Σₜ log P(yₜ | y₁…yₜ₋₁, I)                  │              │
│      │  • Duration: days to weeks on 8-64 GPUs                     │              │
│      └──────────────────────┬─────────────────────────────────────┘              │
│                             │                                                     │
│                             ▼                                                     │
│      ┌──────────────────────────────────────────────────────────────┐             │
│      │  STAGE 2: REINFORCEMENT LEARNING (RLHF / RLVR)              │             │
│      │  ──────────────────────────────────────────────               │             │
│      │  Choose ONE algorithm per run:                                │             │
│      │                                                               │             │
│      │  ┌──────────┐   ┌──────────┐   ┌──────────┐                 │             │
│      │  │   RLHF   │   │   DPO    │   │   GRPO   │                 │             │
│      │  │ (PPO +   │   │ (direct  │   │ (group   │                 │             │
│      │  │  reward  │   │ preference│   │ relative │                 │             │
│      │  │  model)  │   │ optim.)  │   │ policy)  │                 │             │
│      │  └────┬─────┘   └────┬─────┘   └────┬─────┘                 │             │
│      │       │              │              │                         │             │
│      │       ▼              ▼              ▼                         │             │
│      │  θ_A (safe)    θ_B (helpful)   θ_C (precise)                │             │
│      └──────────┬───────────┬──────────────┬────────────────────────┘             │
│                 │           │              │                                       │
│                 ▼           ▼              ▼                                       │
│      ┌──────────────────────────────────────────────────────────────┐             │
│      │  STAGE 3: WEIGHT-SPACE MERGING / MODEL SOUPING               │             │
│      │  ─────────────────────────────────────────────                │             │
│      │  Combine specialized checkpoints in weight space:             │             │
│      │  • Linear interpolation (model soup)                          │             │
│      │  • Task arithmetic: θ = θ_base + Σₖ λₖ·τₖ                    │             │
│      │  • TIES merging (trim + elect sign + disjoint merge)          │             │
│      │  • Cost: minutes, no training, no GPU                         │             │
│      └──────────────────────┬───────────────────────────────────────┘             │
│                             │                                                     │
│                             ▼                                                     │
│                    ┌───────────────────┐                                          │
│                    │   FINAL MODEL     │                                          │
│                    │  (deploy / serve) │                                          │
│                    └───────────────────┘                                          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Supervised Fine-Tuning (SFT) — In Depth

SFT is the foundation of every modern VLM. It trains the model to follow instructions by minimizing cross-entropy on curated input-output pairs.

#### 1.1 The Cross-Entropy Objective — Full Derivation

For a multimodal input consisting of image $I$ and instruction text, the model generates response tokens $y_1, y_2, \ldots, y_T$. The SFT loss decomposes autoregressively via the chain rule of probability:

$$P_\theta(\mathbf{y} \mid I) = \prod_{t=1}^{T} P_\theta(y_t \mid y_1, \ldots, y_{t-1}, I)$$

Taking the negative log-likelihood:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\log P_\theta(\mathbf{y} \mid I) = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_1, \ldots, y_{t-1}, I)
$$

At each timestep $t$, the model produces a logit vector $\mathbf{z}_t \in \mathbb{R}^V$ over the vocabulary of size $V$:

$$P_\theta(y_t = w \mid \cdot) = \frac{e^{z_{t,w}}}{\sum_{w'=1}^{V} e^{z_{t,w'}}}, \qquad \nabla_{z_{t,w}} \mathcal{L} = P_\theta(w \mid \cdot) - \mathbb{1}[w = y_t]$$

The gradient pushes probability mass **toward** the correct token and **away** from all others — softmax cross-entropy is simultaneously attractive and repulsive.

**Why cross-entropy and not MSE?** For classification over $V$ classes, cross-entropy has a gradient that does not vanish when the prediction is confident-but-wrong (softmax always has nonzero gradient), whereas squared error $\|p - \mathbb{1}\|^2$ shrinks near the boundary. This makes cross-entropy strictly preferred for discrete token prediction.

#### 1.2 Selective Loss Masking

In multimodal SFT, you **only compute loss on the response tokens**, not the instruction or image tokens:

```
  Tokens:   <img₁> <img₂> ... <img₅₇₆> [INST] What is this? [/INST] This is a cat.
  Loss:      ✗       ✗    ...    ✗        ✗      ✗     ✗  ✗     ✗     ✓    ✓  ✓ ✓

  L_SFT = -[log P("This"|...) + log P("is"|...) + log P("a"|...) + log P("cat"|...) + log P("."|...)]
```

Mathematically, let $\mathcal{R} \subseteq \lbrace 1, \ldots, T \rbrace$ be the set of response token positions:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{\lvert\mathcal{R}\rvert}\sum_{t \in \mathcal{R}} \log P_\theta(y_t \mid y_1, \ldots, y_{t-1}, I)$$

This prevents the model from wasting capacity learning to predict instruction formatting — it focuses entirely on generating correct responses.

#### 1.3 Data Mixing and Curriculum Learning

Production SFT uses **multiple data sources** with a mixing ratio $\lbrace w_d \rbrace$:

$$\mathcal{L}_{\text{mixed}} = \sum_{d=1}^{D} w_d \cdot \mathbb{E}_{(x,y) \sim \mathcal{D}_d}\left[\mathcal{L}_{\text{SFT}}(x, y)\right]$$

| Data Source | Mix Weight $w_d$ | Purpose |
|-------------|-----------------|---------|
| General caption pairs | 0.30 | Visual grounding |
| OCR / document understanding | 0.25 | Text-in-image reading |
| Instruction following | 0.20 | Chat capabilities |
| Math / reasoning | 0.10 | Chain-of-thought |
| Code with screenshots | 0.10 | UI understanding |
| Safety / refusal | 0.05 | Alignment baseline |

**Curriculum strategy:** Start with simple, high-quality data (captions, short QA), then progressively introduce harder tasks (multi-step reasoning, long documents):

```
  Training progress ──────────────────────────────────────────────▶

  Phase 1 (0-30%):        Phase 2 (30-70%):       Phase 3 (70-100%):
  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
  │ Captions       │      │ Instruction QA │      │ Multi-step     │
  │ Short QA       │      │ OCR documents  │      │ Complex reason.│
  │ Simple desc.   │      │ Tables/charts  │      │ Long documents │
  │                │      │ Spatial tasks  │      │ Multi-image    │
  │ avg_len ≈ 50   │      │ avg_len ≈ 200  │      │ avg_len ≈ 500  │
  └────────────────┘      └────────────────┘      └────────────────┘
        Easy ──────────────── Medium ────────────────── Hard
```

#### 1.4 SFT Hyperparameter Configuration

| Component | Typical Choice | Why | Range |
|-----------|---------------|-----|-------|
| Optimizer | AdamW | Decoupled weight decay; stable for transformers | — |
| $\beta_1, \beta_2$ | 0.9, 0.95 | Lower $\beta_2$ than default — faster adaptation to new data distributions | 0.9–0.999 |
| Weight decay $\lambda$ | 0.1 | Regularization; prevents overfitting to instruction format | 0.01–0.1 |
| Peak LR $\eta_{\max}$ | $1 \times 10^{-5}$ to $2 \times 10^{-5}$ | Lower than pretraining — fine-tuning regime | — |
| Warmup | 1–3% of total steps | Stabilizes early Adam estimates | — |
| LR Schedule | Cosine decay to $\eta_{\min} = 0$ | Smooth annealing; no discrete jumps | — |
| Precision | bf16 + FlashAttention-2 | 50% memory savings; no loss scaling needed | — |
| Batch size | 256–1024 (via gradient accumulation) | Large enough for stable gradients across diverse data | — |
| Max sequence length | 2048–8192 tokens | Varies by task complexity | — |
| Gradient clipping | max\_norm $= 1.0$ | Prevents spikes from long sequences / noisy batches | 0.5–2.0 |
| Epochs | 1–3 | Overfitting is fast on instruction data | — |

#### 1.5 Knowledge Distillation in SFT

Train a smaller student $S$ to mimic a larger teacher $T$. The combined loss balances ground-truth supervision with the teacher's soft probability distribution:

$$
\mathcal{L}_{\text{SFT+KD}} = (1-\alpha)\,\mathcal{L}_{\text{CE}}(S(x), y) + \alpha\, T_{\text{temp}}^2 \,\text{KL}\!\left(\text{softmax}\!\left(\frac{z_S}{T_{\text{temp}}}\right) \;\Big\|\; \text{softmax}\!\left(\frac{z_T}{T_{\text{temp}}}\right)\right)
$$

**Why $T_{\text{temp}}^2$ scaling?** Let $p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$. The gradient of KL w.r.t. student logit $z_i^S$ is:

$$\frac{\partial \text{KL}}{\partial z_i^S} = \frac{1}{T}\left(p_i^S - p_i^T\right)$$

This is scaled by $1/T$, so gradients shrink as $T$ increases. Multiplying by $T^2$ compensates:

$$T^2 \cdot \frac{\partial \text{KL}}{\partial z_i^S} = T\left(p_i^S - p_i^T\right)$$

Now gradients scale proportionally to $T$ rather than inversely — ensuring the teacher's inter-class relationships survive the temperature softening.

**Temperature effect on the teacher's distribution:**

| $T_{\text{temp}}$ | Distribution shape | What student learns |
|-------------------|-------------------|---------------------|
| 1 | Sharp (near one-hot) | Only the top-1 prediction |
| 2–4 | Soft (reveals ranking) | Relative class similarities ("cat" is closer to "dog" than to "car") |
| $\to \infty$ | Uniform ($1/V$) | Nothing useful |

```
  Teacher (235B, frozen)                               Student (1-7B, training)
  ┌──────────────────────┐                             ┌──────────────────────┐
  │   Qwen3-VL-235B      │                             │   Target model       │
  │                      │                             │                      │
  │  Image + Text ───▶ z_T ──── softmax(z_T / T) ─────▶│ KL divergence        │
  │                      │       "soft targets"        │    ▲                  │
  └──────────────────────┘       P_T(y|x)              │    │                  │
                                                        │  z_S ◀── Image+Text  │
  Ground-truth labels y* ────────────────────────────▶│ CE loss              │
                                                        │                      │
                                                        │  Total loss:         │
                                                        │  L = (1-α)·CE(S,y*) │
                                                        │    + α·T²·KL(S‖T)   │
                                                        └──────────────────────┘

  Typical config:
  ┌──────────────────────────────────────────────────────┐
  │  α = 0.5     (equal weight to teacher and labels)     │
  │  T = 4       (soft enough to reveal inter-class info) │
  │  Teacher:    Qwen3-VL-235B (frozen, inference only)   │
  │  Student:    Qwen2.5-VL-7B (all params trainable)     │
  │  Speedup:    Student is ~30× faster at inference       │
  └──────────────────────────────────────────────────────┘
```

**Numerical example** — vocabulary size $V = 5$, teacher logits $z_T = [3.0, 1.0, 0.5, -1.0, -2.0]$:

| Token | $z_T$ | $P_T(T=1)$ | $P_T(T=4)$ | What student sees |
|-------|-------|-----------|-----------|-------------------|
| "cat" | 3.0 | 0.836 | 0.361 | Correct answer |
| "dog" | 1.0 | 0.113 | 0.220 | Semantically close — student learns this! |
| "pet" | 0.5 | 0.069 | 0.196 | Also related |
| "car" | -1.0 | 0.015 | 0.127 | Unrelated |
| "the" | -2.0 | 0.006 | 0.096 | Very unrelated |

At $T=1$, the distribution is 83.6% on "cat" — nearly one-hot. At $T=4$, the student can see that "dog" (22.0%) is much closer to "cat" than "car" (12.7%). This **inter-class structure** is the "dark knowledge" that makes distillation powerful.

---

### Stage 2: Reinforcement Learning — RLHF, DPO, GRPO

After SFT, the model can follow instructions but may still hallucinate, produce formatting errors, or ignore spatial constraints. **RL refines** behavior by optimizing a reward signal beyond what maximum likelihood (SFT) can capture.

```
  Why RL after SFT?

  SFT optimizes:     max P(y* | x)              ← "match the reference answer exactly"
  RL  optimizes:     max E[R(x, y)]             ← "produce ANY answer that scores well"

  SFT limitation:    Many valid outputs, but SFT only sees ONE reference.
                     "A cat sitting on a mat" vs "There is a cat on the mat"
                     → SFT penalizes the second; RL rewards both if R is high.

  RL advantage:      Can optimize non-differentiable metrics (CIDEr, IoU, edit distance)
                     Can incorporate human preferences (safety, helpfulness, style)
```

#### 2a. RLHF (Reinforcement Learning from Human Feedback) — Full Pipeline

RLHF is a **3-step** process: collect preferences, train a reward model, optimize the policy.

**Step 1: Collect human preferences.** For each prompt $x$, sample two responses $y_w$ (preferred / "win") and $y_l$ (rejected / "lose") from the SFT model, then have human annotators rank them: $y_w \succ y_l \mid x$.

**Step 2: Train a reward model** $R_\phi$ using the Bradley-Terry model of pairwise preferences:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)$$

where $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid. The loss is negative log-likelihood:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[\log \sigma\!\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)\right]$$

The gradient w.r.t. $\phi$:

$$\nabla_\phi \mathcal{L}_{\text{RM}} = -\mathbb{E}\left[\sigma\!\left(R_\phi(x, y_l) - R_\phi(x, y_w)\right) \cdot \left(\nabla_\phi R_\phi(x, y_w) - \nabla_\phi R_\phi(x, y_l)\right)\right]$$

When the reward model already assigns higher reward to $y_w$ (correct ordering), $\sigma(R_l - R_w) \approx 0$ → small gradient. When it's wrong, $\sigma(R_l - R_w) \approx 1$ → large correction. The reward model **self-calibrates** its confidence.

**Step 3: Optimize the policy** $\pi_\theta$ using PPO with KL regularization:

$$\max_{\pi_\theta} \;\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[R_\phi(x, y)\right] - \beta\, \text{KL}\!\left(\pi_\theta \;\|\; \pi_{\text{ref}}\right)$$

The KL penalty prevents **reward hacking** — without it, the policy exploits reward model weaknesses (e.g., generating repetitive text that scores high on a flawed $R_\phi$).

**The PPO Objective — Clipped Surrogate:**

Define the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ and the advantage $\hat{A}_t$ (estimated via GAE):

$$\mathcal{L}_{\text{PPO}}^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[\min\!\left(r_t(\theta)\,\hat{A}_t, \;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\right)\right]$$

The clipping ($\epsilon = 0.2$ typically) prevents catastrophically large policy updates:

```
  The PPO clipping mechanism:

  Objective
  L(θ)
    ^
    │          unclipped              clipped (ε=0.2)
    │           ╱                        ┌──────────
    │          ╱                         │
    │         ╱                          │
    │        ╱                           │
    │───────╱────────────────────────────┤
    │      ╱                             │
    │     ╱                              │
    │    ╱                               └──────────
    │   ╱
    +───────────────────────────────────────────────> r(θ)
         0.8    1.0    1.2        0.8    1.0    1.2

  When Â > 0 (good action):        When Â < 0 (bad action):
  • Clip at r = 1+ε (cap reward)    • Clip at r = 1-ε (cap penalty)
  • Prevents over-exploitation       • Prevents over-correction
```

**Generalized Advantage Estimation (GAE):**

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

where $\delta_t$ is the TD residual, $\gamma$ is the discount factor, $\lambda \in [0,1]$ trades bias (low $\lambda$) for variance (high $\lambda$). For language models: $\gamma = 1$ (no discounting within response), $\lambda = 0.95$.

**Value function loss** (the critic):

$$\mathcal{L}_{\text{value}}(\phi) = \frac{1}{2}\mathbb{E}_t\left[(V_\phi(s_t) - V_t^{\text{target}})^2\right]$$

**Full RLHF memory footprint** — 4 models in memory simultaneously:

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    PPO RLHF Memory Layout (7B model)                  │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  1. Policy model π_θ (trainable)           14 GB (bf16)              │
  │  2. Reference model π_ref (frozen)         14 GB (bf16)              │
  │  3. Reward model R_φ (frozen)              14 GB (bf16)              │
  │  4. Value model V_ψ (trainable)            14 GB (bf16)              │
  │  5. Optimizer states (AdamW for π,V)       ~12 GB                    │
  │  6. Activations + KV cache                 ~10 GB                    │
  │  ──────────────────────────────────────────────────                  │
  │  TOTAL                                    ≈ 78 GB → needs 2× A100   │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

**RLHF Hyperparameters:**

| Parameter | Typical Value | Role |
|-----------|-------------|------|
| $\beta$ (KL coefficient) | 0.01–0.1 | Controls policy drift from SFT |
| $\epsilon$ (PPO clip) | 0.2 | Trust region size |
| $\gamma$ (discount) | 1.0 | No discounting for single-turn |
| $\lambda$ (GAE) | 0.95 | Bias-variance trade-off |
| Rollout batch | 512 prompts | Diversity of training signal |
| PPO epochs per batch | 1–4 | Reuse of collected rollouts |
| LR | $1 \times 10^{-6}$ | Very small — RL is unstable |

---

#### 2b. DPO (Direct Preference Optimization) — Full Derivation

DPO is an elegant reformulation that **eliminates the reward model entirely**. The key insight is that the optimal policy under KL-constrained reward maximization has a closed-form solution.

**Starting point:** The RLHF objective:

$$\max_{\pi} \;\mathbb{E}_{y \sim \pi(\cdot \mid x)}\left[r(x,y)\right] - \beta\, \text{KL}(\pi \| \pi_{\text{ref}})$$

**The closed-form optimal policy** (derivation via Lagrangian / calculus of variations):

For each token sequence $y$, the KL-regularized objective is:

$$J(\pi) = \sum_y \pi(y \mid x)\left[r(x,y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right]$$

Setting $\frac{\partial J}{\partial \pi(y \mid x)} = 0$:

$$r(x,y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \beta = 0$$

(The $-\beta$ comes from the entropy term's derivative: $-\beta(1 + \log\pi - \log\pi_{\text{ref}})$.)

Solving for $\pi^*$:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x,y)}{\beta}\right), \quad Z(x) = \sum_{y'} \pi_{\text{ref}}(y' \mid x) \exp\!\left(\frac{r(x,y')}{\beta}\right)$$

**Inverting for the reward:** Rearrange to express $r$ in terms of $\pi^*$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**The DPO trick:** Substitute this into the Bradley-Terry preference model. The $Z(x)$ terms cancel because both $y_w$ and $y_l$ share the same prompt $x$:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(r(x, y_w) - r(x, y_l)\right) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

Now replace $\pi^*$ with the parameterized policy $\pi_\theta$ and maximize log-likelihood:

$$\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]}$$

**Implicit reward:** The trained policy directly defines the reward:

$$r_{\text{DPO}}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \text{const}(x)$$

**Gradient analysis:**

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\left[\underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{scaling}}\left[\underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{increase } y_w} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{decrease } y_l}\right]\right]$$

where $\hat{r}_y = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$.

The scaling factor $\sigma(\hat{r}_l - \hat{r}_w) \in (0,1)$ is large when the model **incorrectly ranks** the pair (assigns higher implicit reward to $y_l$) and small when it already ranks correctly. DPO automatically focuses on hard pairs.

**DPO Numerical Example** ($\beta = 0.1$):

| | $\log \pi_\theta(y \mid x)$ | $\log \pi_{\text{ref}}(y \mid x)$ | $\hat{r}(x,y)$ |
|---|---|---|---|
| $y_w$ (preferred) | $-12.5$ | $-13.0$ | $0.1 \times ((-12.5) - (-13.0)) = 0.05$ |
| $y_l$ (rejected) | $-14.2$ | $-13.8$ | $0.1 \times ((-14.2) - (-13.8)) = -0.04$ |

$$\Delta = \hat{r}_w - \hat{r}_l = 0.05 - (-0.04) = 0.09$$

$$\mathcal{L}_{\text{DPO}} = -\log\sigma(0.09) = -\log(0.5225) = 0.649$$

The loss is moderately high because $\Delta = 0.09$ is small — the model barely prefers $y_w$ over $y_l$. Training will push $\pi_\theta$ to increase $P(y_w)$ and decrease $P(y_l)$.

**DPO Memory — Only 2 models:**

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    DPO Memory Layout (7B model)                   │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  1. Policy model π_θ (trainable)         14 GB (bf16)            │
  │  2. Reference model π_ref (frozen)       14 GB (bf16)            │
  │  3. Optimizer states (AdamW)             ~6 GB                   │
  │  4. Activations                          ~4 GB                   │
  │  ──────────────────────────────────────────────                  │
  │  TOTAL                                  ≈ 38 GB → 1× A100       │
  │                                                                  │
  │  With LoRA: π_ref = frozen base inside PEFT model (FREE!)       │
  │  TOTAL with LoRA                        ≈ 18 GB → 1× RTX 4090  │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

---

#### 2c. GRPO (Group Relative Policy Optimization) — Full Algorithm

GRPO (DeepSeek-R1, 2024) eliminates both the reward model AND the value model by using **group-relative advantages** from verifiable rewards.

**Core idea:** For each prompt $x$, generate a **group** of $G$ responses. Score each with a deterministic, verifiable reward function $R(x, y_i)$. Normalize rewards within the group to get advantages — no learned critic needed.

**Step 1 — Group sampling and scoring:**

$$\lbrace y_1, y_2, \ldots, y_G \rbrace \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$$

$$R_i = R(x, y_i) \in \mathbb{R}, \quad i = 1, \ldots, G$$

**Step 2 — Group-relative advantage:**

$$A_i = \frac{R_i - \mu_G}{\sigma_G + \epsilon}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^{G} R_j, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j - \mu_G)^2}$$

This normalization ensures $\mathbb{E}[A_i] \approx 0$ and $\text{Var}(A_i) \approx 1$ — stabilizing the policy gradient without a learned baseline.

**Step 3 — Clipped policy gradient with KL regularization:**

Let $r_i(\theta) = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$ be the importance sampling ratio. The GRPO loss is:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \min\!\left(r_i(\theta)\, A_i,\;\text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\, A_i\right) + \beta\,\text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

The KL term is computed per-token and averaged:

$$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \frac{1}{T}\sum_{t=1}^{T}\left[\frac{\pi_\theta(y_t \mid y_1, \ldots, y_{t-1}, x)}{\pi_{\text{ref}}(y_t \mid y_1, \ldots, y_{t-1}, x)} - \log\frac{\pi_\theta(y_t \mid \cdot)}{\pi_{\text{ref}}(y_t \mid \cdot)} - 1\right]$$

**Verifiable Reward Functions** — the key enabler for GRPO:

| Task | Reward Function $R(x, y)$ | Formula | Auto-gradeable? |
|------|--------------------------|---------|----------------|
| OCR | Character Error Rate (CER) | $R = 1 - \frac{\text{edit\_dist}(y, y^*)}{\max(\lvert y \rvert, \lvert y^* \rvert)}$ | Yes |
| Math | Exact answer match | $R = \mathbb{1}[\text{extract}(y) = a^*]$ | Yes |
| Bounding box | Intersection over Union | $R = \text{IoU}(\text{bbox}(y), \text{bbox}^*)$ | Yes |
| Code | Test suite pass rate | $R = \frac{\text{tests\_passed}}{\text{total\_tests}}$ | Yes |
| Format compliance | Regex / schema match | $R = \mathbb{1}[\text{matches\_schema}(y)]$ | Yes |
| Anti-repetition | N-gram penalty | $R_{\text{rep}} = -\lambda \cdot \frac{\text{repeated\_ngrams}}{\text{total\_ngrams}}$ | Yes |

**Composite reward** (combine multiple signals):

$$R_{\text{total}}(x, y) = w_1 R_{\text{task}}(x, y) + w_2 R_{\text{format}}(x, y) + w_3 R_{\text{rep}}(x, y)$$

**GRPO Step-by-Step Numerical Example** ($G=4$, $\epsilon=0.2$, $\beta=0.01$):

Given prompt $x$ = "OCR this receipt image", the model generates 4 responses:

| Response $y_i$ | Predicted text | CER | $R_i$ | Format bonus | $R_{\text{total}}$ |
|----------------|---------------|-----|-------|-------------|-------------------|
| $y_1$ | "Total: \$42.50" | 0.00 | 1.00 | +0.1 (correct format) | **1.10** |
| $y_2$ | "Total: \$42.5O" | 0.07 | 0.93 | +0.1 | **1.03** |
| $y_3$ | "Totla: \$42.50" | 0.07 | 0.93 | +0.1 | **1.03** |
| $y_4$ | "Total 42.50 total 42.50" | 0.40 | 0.60 | -0.2 (repetition) | **0.40** |

**Compute advantages:**

$$\mu_G = \frac{1.10 + 1.03 + 1.03 + 0.40}{4} = 0.89, \quad \sigma_G = \sqrt{\frac{(0.21)^2 + (0.14)^2 + (0.14)^2 + (-0.49)^2}{4}} = 0.276$$

| | $R_i$ | $R_i - \mu$ | $A_i = \frac{R_i - \mu}{\sigma}$ | Interpretation |
|---|---|---|---|---|
| $y_1$ | 1.10 | +0.21 | **+0.76** | Above average → reinforce |
| $y_2$ | 1.03 | +0.14 | **+0.51** | Slightly above → mild reinforce |
| $y_3$ | 1.03 | +0.14 | **+0.51** | Slightly above → mild reinforce |
| $y_4$ | 0.40 | -0.49 | **-1.78** | Far below → strongly suppress |

The repetitive $y_4$ gets a strongly negative advantage (-1.78), meaning the policy gradient will decrease its probability. The perfect $y_1$ gets moderate positive advantage (+0.76).

**GRPO Training Loop — Complete ASCII Diagram:**

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        GRPO TRAINING LOOP                                     │
│                                                                               │
│  for each batch of prompts {x₁, x₂, ..., x_B}:                             │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: ROLLOUT GENERATION (via vLLM for speed)                       │  │
│  │                                                                         │  │
│  │  for each xₖ:                                                          │  │
│  │    Generate G responses: {y₁, y₂, ..., y_G} ~ π_θ_old(·|xₖ)          │  │
│  │    Store log-probs: log π_θ_old(yᵢ|xₖ) for importance sampling        │  │
│  │                                                                         │  │
│  │  Total: B × G responses  (e.g., 64 prompts × 8 rollouts = 512)        │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: REWARD SCORING (deterministic, no gradient)                    │  │
│  │                                                                         │  │
│  │  for each (xₖ, yᵢ):                                                   │  │
│  │    Rᵢ = w₁·R_task(xₖ,yᵢ) + w₂·R_format(xₖ,yᵢ) + w₃·R_rep(xₖ,yᵢ)   │  │
│  │                                                                         │  │
│  │  All rewards computed WITHOUT backprop (fast, parallelizable)           │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: ADVANTAGE COMPUTATION (per-group normalization)                │  │
│  │                                                                         │  │
│  │  for each prompt xₖ:                                                   │  │
│  │    μₖ = mean({Rᵢ}ᵢ₌₁ᴳ)                                               │  │
│  │    σₖ = std({Rᵢ}ᵢ₌₁ᴳ)                                                │  │
│  │    Aᵢ = (Rᵢ - μₖ) / (σₖ + ε)      for i = 1..G                      │  │
│  │                                                                         │  │
│  │  Key: normalization is WITHIN each group, not across batch              │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: POLICY GRADIENT UPDATE (with gradient)                         │  │
│  │                                                                         │  │
│  │  for each (xₖ, yᵢ, Aᵢ):                                              │  │
│  │    Compute: log π_θ(yᵢ|xₖ)  (current policy, requires grad)           │  │
│  │    Ratio:   rᵢ = exp(log π_θ - log π_θ_old)                           │  │
│  │    Clipped: L = -min(rᵢ·Aᵢ, clip(rᵢ,1±ε)·Aᵢ) + β·KL(π_θ‖π_ref)    │  │
│  │                                                                         │  │
│  │  Backward + AdamW step                                                  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 5: UPDATE OLD POLICY                                              │  │
│  │                                                                         │  │
│  │  θ_old ← θ   (sync for next round of rollouts)                        │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  Repeat until convergence (typically 1-3 epochs over the prompt dataset)     │
└───────────────────────────────────────────────────────────────────────────────┘
```

**GRPO Hyperparameters:**

| Parameter | Typical Value | Role |
|-----------|-------------|------|
| Group size $G$ | 8–32 | More rollouts → better advantage estimation |
| $\epsilon$ (clip) | 0.2 | Trust region radius |
| $\beta$ (KL coeff) | 0.001–0.05 | Prevents drift from reference |
| Learning rate | $1 \times 10^{-6}$ to $5 \times 10^{-6}$ | Very small for stability |
| Rollout temperature | 0.7–1.0 | Controls diversity of samples |
| Max response length | 2048–8192 tokens | Task-dependent |
| Batch size $B$ | 64–256 prompts | × $G$ rollouts each |
| PPO epochs | 1 (single-epoch is most stable) | Reuse within batch |

**GRPO Memory — Only 1 trainable model:**

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    GRPO Memory Layout (7B model)                   │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  1. Policy model π_θ (trainable)         14 GB (bf16)            │
  │  2. Reference model π_ref (frozen)       14 GB (bf16)            │
  │     (or: use KV cache snapshots to avoid full copy)              │
  │  3. Rollout cache (G responses)          ~2 GB (text only)       │
  │  4. Optimizer states (AdamW)             ~6 GB                   │
  │  5. Activations                          ~4 GB                   │
  │  ──────────────────────────────────────────────                  │
  │  TOTAL                                  ≈ 40 GB → 1× A100       │
  │                                                                  │
  │  With QLoRA: base NF4 + LoRA fp16                                │
  │  TOTAL with QLoRA                       ≈ 12 GB → 1× RTX 4090  │
  │                                                                  │
  │  Note: vLLM handles rollout generation separately                │
  │  (can use a different GPU or batched inference server)            │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

---

#### RL Algorithm Comparison — Complete Analysis

| Property | **RLHF (PPO)** | **DPO** | **GRPO** | **REINFORCE** |
|----------|----------------|---------|----------|---------------|
| **Reward model** | Required (learned $R_\phi$) | Not needed | Not needed (verifiable) | Required |
| **Value model** | Required (learned $V_\psi$) | Not needed | Not needed | Not needed |
| **Models in memory** | 4 (π, π_ref, R, V) | 2 (π, π_ref) | 2 (π, π_ref) | 2 (π, R) |
| **Memory (7B, bf16)** | ~78 GB | ~38 GB | ~40 GB | ~32 GB |
| **Stability** | Moderate (many hyperparams) | High (simple loss) | High (self-normalizing) | Low (high variance) |
| **Training data** | Preference pairs | Preference pairs | Prompts + reward fn | Prompts + reward fn |
| **Human labels** | Yes (for reward model) | Yes (for pairs) | No (verifiable rewards) | Yes (for reward model) |
| **Reward hacking risk** | Moderate (KL helps) | Low (implicit reward) | Low (verifiable) | High |
| **Online/Offline** | Online (generates) | Offline (fixed pairs) | Online (generates) | Online |
| **Best for** | General alignment | Preference tuning | Tasks with auto-grading | Simple baselines |
| **Key papers** | InstructGPT (2022) | Rafailov et al. (2023) | DeepSeek-R1 (2024) | Williams (1992) |

**When to choose which:**

```
  Do you have human preference data?
       │
       ├── YES ──▶ Is it expensive to collect more?
       │               │
       │               ├── YES ──▶ DPO (offline, uses existing pairs efficiently)
       │               │
       │               └── NO ──▶ RLHF/PPO (online, explores with reward model)
       │
       └── NO ──▶ Can you auto-grade responses?
                       │
                       ├── YES ──▶ GRPO (verifiable rewards, no human needed)
                       │
                       └── NO ──▶ Collect human labels first, then DPO
```

---

### Stage 3: Weight-Space Merging (Model Souping) — In Depth

After training multiple specialized models via RL (e.g., one optimized for OCR, another for bounding boxes), **weight merging** combines their strengths without additional training. This is the cheapest and most underrated stage — minutes of CPU computation can achieve Pareto-optimal multi-task performance.

#### Why Merging Works: The Loss Basin Hypothesis

Fine-tuned models from the same base often lie in the **same loss basin** of the loss landscape. Linear paths between them stay in low-loss regions:

```
  Loss L(θ)
    ^
    │    ╲          ╱
    │     ╲  high  ╱
    │      ╲ loss ╱
    │       ╲    ╱
    │        ╲  ╱
    │    θ_A  ╲╱  θ_B           Different base → different basin
    │    ●────────●                → merging fails (high loss on path)
    │     ╲  low ╱
    │      ╲loss╱
    │       ╲  ╱
    │        ●
    │     θ_merged                  Same base → same basin
    │   (linear path               → merging succeeds (low loss on path)
    │    stays in basin)
    └──────────────────────────────> θ
```

**Sufficient condition:** Models must be fine-tuned from the **same base checkpoint** $\theta^{\text{base}}$ (same random seed, same pretrained weights). This ensures they're in the same basin.

#### Method 1: Linear Interpolation (Model Soup)

The simplest merging method — weighted average of full checkpoints:

$$\theta_{\text{merged}} = \sum_{k=1}^{K} \alpha_k \, \theta_k, \qquad \sum_{k=1}^{K} \alpha_k = 1$$

where $\theta_k$ are checkpoints from different RL runs or different epochs.

**Greedy soup construction** (Wortsman et al., 2022): Instead of averaging all models, start with the best and greedily add others only if they improve validation accuracy:

```
  Greedy Model Soup Algorithm:

  1. Sort models by individual validation accuracy: θ₁ ≥ θ₂ ≥ ... ≥ θ_K
  2. Initialize soup: θ_soup = θ₁
  3. For k = 2, ..., K:
     a. Candidate: θ_cand = (k-1)/(k) · θ_soup + 1/k · θ_k     (uniform avg)
     b. If acc(θ_cand) > acc(θ_soup):
        θ_soup ← θ_cand    (add to soup)
     c. Else: skip θ_k       (would hurt performance)
  4. Return θ_soup
```

**Numerical example** ($K = 3$ checkpoints, 2 tasks):

| Model | OCR acc | BBox mAP | Avg |
|-------|---------|----------|-----|
| $\theta_1$ (OCR-tuned) | **92.1%** | 78.3% | 85.2% |
| $\theta_2$ (BBox-tuned) | 84.5% | **91.7%** | 88.1% |
| $\theta_3$ (balanced) | 88.0% | 85.2% | 86.6% |
| $\frac{1}{3}(\theta_1 + \theta_2 + \theta_3)$ — uniform soup | 89.8% | 87.4% | **88.6%** |
| $0.4\theta_1 + 0.4\theta_2 + 0.2\theta_3$ — weighted | **90.2%** | **88.1%** | **89.2%** |

The weighted soup exceeds every individual model on the average metric — this is the power of ensembling in weight space.

#### Method 2: Task Arithmetic

Define a **task vector** as the delta from the base model to the fine-tuned model:

$$\tau_k = \theta_k^{\text{fine}} - \theta^{\text{base}}$$

Task vectors are **composable** — merge by adding scaled vectors to the base:

$$\theta_{\text{merged}} = \theta^{\text{base}} + \sum_{k=1}^{K} \lambda_k \, \tau_k$$

**Three operations with task vectors:**

```
  1. ADDITION — compose capabilities:
     θ = θ_base + λ₁·τ_OCR + λ₂·τ_BBox
     → model gains BOTH OCR and BBox skills

  2. NEGATION — remove capabilities:
     θ = θ_base - λ·τ_toxic
     → model LOSES toxic generation ability
     (Ilharco et al., 2023: "Editing Models with Task Arithmetic")

  3. ANALOGY — transfer between domains:
     τ_target = τ_source_A→B + (θ_target_A - θ_base)
     → if you know how to go A→B for source domain,
       apply the same delta to a different model
```

**Mathematical justification:** If the loss landscape is approximately quadratic near $\theta^{\text{base}}$ (valid for small task vectors):

$$\mathcal{L}(\theta^{\text{base}} + \sum_k \lambda_k \tau_k) \approx \mathcal{L}(\theta^{\text{base}}) + \sum_k \lambda_k \nabla\mathcal{L}^\top \tau_k + \frac{1}{2}\sum_{j,k}\lambda_j \lambda_k \tau_j^\top H \tau_k$$

When task vectors are **nearly orthogonal** ($\tau_j^\top H \tau_k \approx 0$ for $j \neq k$), the cross-terms vanish and each task vector independently reduces its task's loss. This is why merging works best when tasks are diverse (OCR vs. spatial reasoning vs. safety).

**Choosing $\lambda_k$:** Grid search on a small validation set. Typical range: $\lambda_k \in [0.1, 1.5]$. Values $> 1.0$ overshoot the fine-tuned solution (sometimes beneficial).

#### Method 3: TIES Merging (Trim, Elect Sign, Disjoint Merge)

TIES (Yadav et al., 2023) addresses **task vector interference** — when two task vectors modify the same parameter in opposite directions, simple averaging cancels them out.

**The three steps:**

**Step 1 — Trim:** Keep only the top-$k\%$ largest-magnitude entries in each task vector. Small changes are noise that causes interference.

$$
\tilde{\tau}_{k}^{(j)} = \begin{cases} \tau_k^{(j)} & \text{if } \lvert\tau_k^{(j)}\rvert \geq \text{quantile}(\lvert\tau_k\rvert, 1-p) \\ 0 & \text{otherwise} \end{cases}
$$

**Step 2 — Elect sign:** For each parameter $j$, take a majority vote of the non-zero trimmed values:

$$s^{(j)} = \text{sign}\!\left(\sum_{k=1}^{K} \tilde{\tau}_k^{(j)}\right)$$

**Step 3 — Disjoint merge:** Average only the trimmed values that agree with the elected sign:

$$\mathcal{A}_j = \lbrace k : \text{sign}(\tilde{\tau}_k^{(j)}) = s^{(j)} \rbrace$$

$$\theta_{\text{merged}}^{(j)} = \theta_{\text{base}}^{(j)} + \lambda \cdot \frac{1}{\lvert\mathcal{A}_j\rvert}\sum_{k \in \mathcal{A}_j} \tilde{\tau}_k^{(j)}$$

**Full TIES Numerical Example** (3 task vectors, 6 parameters, $p = 0.5$ keep top 50%):

```
  Task vectors (raw):
  τ_A = [+0.8, -0.3, +0.1, -0.7, +0.02, +0.5]
  τ_B = [+0.6, +0.4, -0.2, -0.5, +0.01, -0.3]
  τ_C = [-0.1, -0.2, +0.9, +0.3, -0.6,  +0.4]

  Step 1: TRIM (keep top 50% by magnitude per vector):
  ┌──────────┬───────────────────────────────────────┬──────────────────────────────────────┐
  │ Vector   │ Magnitudes                             │ Trimmed (keep top 3)                  │
  ├──────────┼───────────────────────────────────────┼──────────────────────────────────────┤
  │ τ_A      │ [0.8, 0.3, 0.1, 0.7, 0.02, 0.5]      │ [+0.8,  0, 0, -0.7,  0, +0.5]       │
  │ τ_B      │ [0.6, 0.4, 0.2, 0.5, 0.01, 0.3]      │ [+0.6, +0.4, 0, -0.5,  0,  0]       │
  │ τ_C      │ [0.1, 0.2, 0.9, 0.3, 0.6, 0.4]       │ [ 0,    0, +0.9,  0, -0.6, +0.4]    │
  └──────────┴───────────────────────────────────────┴──────────────────────────────────────┘

  Step 2: ELECT SIGN (majority vote per parameter):
  ┌────────┬───────────┬──────────┬──────────┬─────────────────────────────────────┐
  │ Param  │ τ̃_A       │ τ̃_B      │ τ̃_C      │ Sum → Elected sign                   │
  ├────────┼───────────┼──────────┼──────────┼─────────────────────────────────────┤
  │ j=1    │ +0.8      │ +0.6     │  0       │ +1.4 → s=+                           │
  │ j=2    │  0        │ +0.4     │  0       │ +0.4 → s=+                           │
  │ j=3    │  0        │  0       │ +0.9     │ +0.9 → s=+                           │
  │ j=4    │ -0.7      │ -0.5     │  0       │ -1.2 → s=−                           │
  │ j=5    │  0        │  0       │ -0.6     │ -0.6 → s=−                           │
  │ j=6    │ +0.5      │  0       │ +0.4     │ +0.9 → s=+                           │
  └────────┴───────────┴──────────┴──────────┴─────────────────────────────────────┘

  Step 3: DISJOINT MERGE (avg values matching elected sign):
  ┌────────┬──────────┬───────────────────────────────────────────────────────────┐
  │ Param  │ Sign     │ Agreeing values → merged Δ                                │
  ├────────┼──────────┼───────────────────────────────────────────────────────────┤
  │ j=1    │ +        │ {+0.8, +0.6}       → avg = +0.70                          │
  │ j=2    │ +        │ {+0.4}             → avg = +0.40                          │
  │ j=3    │ +        │ {+0.9}             → avg = +0.90                          │
  │ j=4    │ −        │ {-0.7, -0.5}       → avg = -0.60                          │
  │ j=5    │ −        │ {-0.6}             → avg = -0.60                          │
  │ j=6    │ +        │ {+0.5, +0.4}       → avg = +0.45                          │
  └────────┴──────────┴───────────────────────────────────────────────────────────┘

  Final: θ_merged = θ_base + λ · [+0.70, +0.40, +0.90, -0.60, -0.60, +0.45]
```

Notice how parameter $j=4$ had conflicting signs in $\tau_A$ (negative) and $\tau_C$ (positive) — TIES resolves this by majority vote and only averaging agreeing values. Simple averaging would have produced $(-0.7 - 0.5 + 0.3)/3 = -0.3$ which partially cancels the signal.

#### Method 4: DARE Merging (Drop And REscale)

DARE (Yu et al., 2024) randomly drops entries in each task vector and rescales to maintain expectation:

$$
\tilde{\tau}_k^{(j)} = \begin{cases} \frac{\tau_k^{(j)}}{1 - p_{\text{drop}}} & \text{with probability } 1 - p_{\text{drop}} \\ 0 & \text{with probability } p_{\text{drop}} \end{cases}
$$

The $\frac{1}{1-p_{\text{drop}}}$ rescaling ensures $\mathbb{E}[\tilde{\tau}_k^{(j)}] = \tau_k^{(j)}$ — same mean as the original, but sparser. After dropping, merge with any method (linear, task arithmetic, TIES):

$$\theta_{\text{merged}} = \theta^{\text{base}} + \sum_k \lambda_k \tilde{\tau}_k$$

**Intuition:** Redundant parameters create interference. Random dropout breaks correlations between task vectors, similar to how dropout regularizes neural networks.

**DARE + TIES** is the current SOTA for multi-task merging — DARE sparsifies, then TIES resolves sign conflicts.

#### Merging Method Comparison

| Method | Interference handling | Hyperparameters | Quality | Cost |
|--------|----------------------|-----------------|---------|------|
| **Linear (Soup)** | None (naive avg) | $\alpha_k$ weights | Good | Seconds |
| **Task Arithmetic** | None | $\lambda_k$ per task | Good | Seconds |
| **TIES** | Trim + sign election | $p$ (trim%), $\lambda$ | Better | Seconds |
| **DARE** | Random dropout | $p_{\text{drop}}$, $\lambda$ | Better | Seconds |
| **DARE + TIES** | Dropout + trim + sign | $p_{\text{drop}}$, $p_{\text{trim}}$, $\lambda$ | **Best** | Seconds |

```
  Weight-Space Merging — Full Visualization:

  Base Model θ_base
       │
       ├── SFT → θ_SFT
       │     │
       │     ├── DPO (safety data)      → θ_safe     → τ_safe = θ_safe - θ_SFT
       │     ├── DPO (helpfulness data) → θ_helpful   → τ_helpful = θ_helpful - θ_SFT
       │     ├── GRPO (OCR rewards)     → θ_ocr       → τ_ocr = θ_ocr - θ_SFT
       │     └── GRPO (bbox rewards)    → θ_bbox      → τ_bbox = θ_bbox - θ_SFT
       │
       │                    ┌─────────────────────────────────────────────┐
       │                    │           MERGING STRATEGIES                 │
       │                    │                                             │
       │                    │  Linear:                                     │
       │                    │    θ = 0.3·θ_safe + 0.3·θ_helpful           │
       │                    │        + 0.2·θ_ocr + 0.2·θ_bbox            │
       │                    │                                             │
       │                    │  Task Arithmetic:                            │
       │                    │    θ = θ_SFT + 0.7·τ_safe + 0.5·τ_helpful  │
       │                    │        + 0.8·τ_ocr + 0.6·τ_bbox            │
       │                    │                                             │
       │                    │  DARE + TIES:                                │
       │                    │    1. DARE: randomly drop 90% of each τₖ    │
       │                    │    2. TIES: trim → elect sign → merge       │
       │                    │    3. θ = θ_SFT + λ·merged_vector           │
       │                    │                                             │
       │                    └──────────────────┬──────────────────────────┘
       │                                       │
       │                                       ▼
       │                              θ_merged (ALL capabilities)
       │                              ─────────────────────────
       │                              Safe + helpful + OCR + BBox
       │                              Single model, zero extra cost
       │
       └── Deploy: torch.compile / vLLM / ONNX
```

---

### End-to-End Example 1: LLaVA-Style 3-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  LLaVA-Style Full Training Pipeline                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: SUPERVISED PRETRAINING (SFT — Phase 1: Alignment)                 │
│  ═══════════════════════════════════════════════════════════                  │
│  Goal:     Align vision encoder to LLM's embedding space                     │
│  Dataset:  595K image-caption pairs (CC3M filtered)                          │
│  Frozen:   CLIP ViT-L/14 (vision) + Vicuna-7B (language)                    │
│  Trainable: MLP projector only (~21M params)                                 │
│  Objective: Next-token prediction (cross-entropy)                            │
│  Config:   AdamW, lr=1e-3, cosine decay, bf16, batch=256                    │
│  Duration: ~24 hours on 8× A100                                             │
│  Loss:     L = -Σₜ log P(yₜ | y₁…yₜ₋₁, MLP(ViT(I)))                      │
│                                                                              │
│  ┌────────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐      │
│  │  Image I   │──▶│ CLIP ViT │──▶│ MLP Projector│──▶│   Vicuna-7B   │      │
│  └────────────┘    │ (frozen) │    │ (TRAINING)  │    │   (frozen)    │      │
│                    └──────────┘    └─────────────┘    └──────┬───────┘      │
│                                                              │               │
│                    Caption tokens ──────────────────────────▶│               │
│                                                              ▼               │
│                                                     L_CE(predicted, target)  │
│                                                                              │
│  Output: → LLaVA-base (projector aligned, LLM untouched)                    │
│                   │                                                          │
│                   ▼                                                          │
│  Stage 1.5: INSTRUCTION TUNING (SFT — Phase 2: Full Tuning)                 │
│  ═══════════════════════════════════════════════════════════                  │
│  Goal:     Teach model to follow multimodal instructions                     │
│  Dataset:  150K instruction-following conversations (LLaVA-Instruct)         │
│             Format: <image> + "Describe this image in detail" → response     │
│  Frozen:   CLIP ViT-L/14 only                                               │
│  Trainable: MLP projector + LoRA on LLM (r=128, α=256)                     │
│  Objective: Next-token prediction on RESPONSE tokens only                    │
│  Config:   AdamW, lr=2e-5, cosine decay, bf16, batch=128                    │
│  Duration: ~12 hours on 8× A100                                             │
│                                                                              │
│  ┌────────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐      │
│  │  Image I   │──▶│ CLIP ViT │──▶│ MLP Projector│──▶│  Vicuna-7B   │      │
│  └────────────┘    │ (frozen) │    │ (TRAINING)  │    │  + LoRA      │      │
│                    └──────────┘    └─────────────┘    │  (TRAINING)  │      │
│                                                        └──────┬───────┘      │
│  [INST] tokens ──────────────────────────────────────────────▶│              │
│  (loss masked)                                                 │              │
│                                                                ▼              │
│  Response tokens ─────────────────────────────────── L_CE (loss computed)    │
│                                                                              │
│  Output: → LLaVA-v1.5-7B (instruction-following VLM)                        │
│                   │                                                          │
│                   ▼                                                          │
│  Stage 2: DPO ALIGNMENT                                                      │
│  ═══════════════════════                                                      │
│  Goal:     Reduce hallucinations, improve factual grounding to image         │
│  Algorithm: DPO (β = 0.1), no reward model needed                            │
│  Data:     10K preference pairs (y_w, y_l) ranked by human annotators        │
│            • y_w: factually correct, well-grounded                           │
│            • y_l: hallucinated objects, wrong spatial relations               │
│  Frozen:   Vision encoder                                                    │
│  Trainable: LoRA on LLM (r=64) + projector                                 │
│  Config:   AdamW, lr=5e-7, β=0.1, bf16, batch=32                           │
│                                                                              │
│  Train TWO variants for merging:                                             │
│                                                                              │
│  ┌─────────────────────────┐         ┌─────────────────────────┐            │
│  │ Variant A: Safety DPO   │         │ Variant B: Quality DPO   │            │
│  │                         │         │                         │            │
│  │ Preference pairs:       │         │ Preference pairs:       │            │
│  │ y_w: refuses harmful    │         │ y_w: detailed, accurate │            │
│  │ y_l: complies unsafely  │         │ y_l: vague, incomplete  │            │
│  │                         │         │                         │            │
│  │ L = -log σ(β·Δlog)     │         │ L = -log σ(β·Δlog)     │            │
│  └──────────┬──────────────┘         └──────────┬──────────────┘            │
│             │                                   │                            │
│             ▼                                   ▼                            │
│  θ_safe (τ_safe = θ_safe - θ_SFT)   θ_helpful (τ_helpful = θ_helpful - θ_SFT)│
│             │                                   │                            │
│             └───────────────┬───────────────────┘                            │
│                             ▼                                                │
│  Stage 3: WEIGHT MERGING                                                     │
│  ═══════════════════════════                                                  │
│  Method:   Task arithmetic with validation-tuned λ                           │
│  Formula:  θ_merged = θ_SFT + 0.7·τ_safe + 0.5·τ_helpful                   │
│  Tuning:   Grid search λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} on 500 val samples   │
│  Cost:     ~5 minutes on CPU (just vector addition)                          │
│                                                                              │
│  Final benchmarks:                                                           │
│  ┌────────────────────┬──────────┬───────────┬─────────────┬──────────┐     │
│  │ Model              │ MMMU     │ POPE (↓H) │ LLaVA-Bench │ Safety   │     │
│  ├────────────────────┼──────────┼───────────┼─────────────┼──────────┤     │
│  │ LLaVA-SFT          │ 35.2     │ 82.1      │ 68.4        │ 72.0     │     │
│  │ LLaVA-safe         │ 33.8     │ 88.5      │ 64.1        │ 91.2     │     │
│  │ LLaVA-helpful      │ 36.1     │ 83.0      │ 74.8        │ 73.5     │     │
│  │ LLaVA-merged       │ 35.6     │ 87.2      │ 72.5        │ 88.1     │     │
│  └────────────────────┴──────────┴───────────┴─────────────┴──────────┘     │
│                                                                              │
│  Output: → LLaVA-v1.5-7B-merged (safe + helpful + capable)                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### End-to-End Example 2: OCR-Specialized VLM Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              OCR-Specialized VLM — Full 3-Stage Pipeline                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: SUPERVISED PRETRAINING (SFT + Knowledge Distillation)              │
│  ═══════════════════════════════════════════════════════════════              │
│  Goal:     Build foundational OCR and document understanding                 │
│  Dataset:  43M pages across 12 categories:                                   │
│            ┌─────────────────┬──────────┬────────────────────┐              │
│            │ Category        │ Samples  │ Format             │              │
│            ├─────────────────┼──────────┼────────────────────┤              │
│            │ Scientific PDFs │ 12M      │ LaTeX ground truth │              │
│            │ Book pages      │ 8M       │ OCR + layout       │              │
│            │ Receipts        │ 5M       │ Key-value pairs    │              │
│            │ Invoices        │ 4M       │ Table structure    │              │
│            │ Handwriting     │ 3M       │ Text transcription │              │
│            │ Tables          │ 3M       │ HTML/Markdown      │              │
│            │ Forms           │ 2M       │ Field extraction   │              │
│            │ Screenshots     │ 2M       │ UI element parsing │              │
│            │ Charts          │ 1.5M     │ Data extraction    │              │
│            │ Maps            │ 1M       │ Label reading      │              │
│            │ Slides          │ 1M       │ Content extraction │              │
│            │ Biz cards       │ 0.5M     │ Contact parsing    │              │
│            └─────────────────┴──────────┴────────────────────┘              │
│                                                                              │
│  Teacher:  Qwen3-VL-235B (frozen, knowledge distillation)                   │
│  Student:  Qwen2.5-VL-7B (all params trainable except ViT)                 │
│  Objective:                                                                  │
│    L = (1-α)·CE(student, y*) + α·T²·KL(softmax(z_S/T) ‖ softmax(z_T/T))  │
│    α = 0.5, T = 4                                                           │
│  Config:   AdamW (β₁=0.9, β₂=0.95, λ=0.1)                                │
│            LR: 2e-5 with cosine decay + 1000-step warmup                    │
│            bf16 + FlashAttention-2                                           │
│            Batch: 384 (via gradient accumulation: 48 × 8 GPUs)              │
│            Gradient clipping: max_norm = 1.0                                 │
│  Duration: ~7 days on 64× A100                                              │
│                                                                              │
│  ┌──────────────┐    ┌──────────┐    ┌─────────────┐                        │
│  │ Qwen3-VL-235B│    │ Student  │    │ Ground Truth│                        │
│  │ (Teacher)    │    │ 7B model │    │ labels y*   │                        │
│  └──────┬───────┘    └────┬─────┘    └──────┬──────┘                        │
│         │                 │                  │                                │
│         │ soft targets    │ z_S              │ y*                             │
│         │ P_T(y|x)       │                  │                                │
│         └────────┐  ┌────┘                  │                                │
│                  ▼  ▼                       ▼                                │
│            L = α·T²·KL(S‖T) + (1-α)·CE(S, y*)                              │
│                                                                              │
│  Outputs:  → OCR-VLM-base       (general OCR)                               │
│            → OCR-VLM-bbox-base  (with bounding box detection head)           │
│                  │                                                            │
│                  ▼                                                            │
│  Stage 2: REINFORCEMENT LEARNING (RLVR via GRPO)                             │
│  ══════════════════════════════════════════════════                            │
│  Goal:     Sharpen OCR accuracy and bbox precision beyond SFT ceiling        │
│  Algorithm: GRPO (Group Relative Policy Optimization)                         │
│  KL reg:   β = 0.01                                                          │
│  Tooling:  HuggingFace TRL + vLLM (rollout generation)                      │
│  Config:   AdamW, lr=1e-6, ε_clip=0.2, bf16                                │
│            Gradient accumulation: 4 steps                                    │
│                                                                              │
│  ┌──────────────────────────────┐    ┌─────────────────────────────────┐    │
│  │  OCR-focused RLVR            │    │  BBox-focused RLVR              │    │
│  │  ───────────────────          │    │  ────────────────────           │    │
│  │  Rollouts: G = 28 per prompt │    │  Rollouts: G = 14 per prompt   │    │
│  │                               │    │                                 │    │
│  │  Reward function:             │    │  Reward function:               │    │
│  │  ┌─────────────────────────┐ │    │  ┌─────────────────────────┐   │    │
│  │  │ R = 0.5·(1-CER)        │ │    │  │ R = 0.4·mean_IoU        │   │    │
│  │  │   + 0.3·format_score   │ │    │  │   + 0.3·ID_overlap      │   │    │
│  │  │   + 0.2·anti_rep       │ │    │  │   + 0.2·count_accuracy  │   │    │
│  │  │                        │ │    │  │   + 0.1·anti_rep        │   │    │
│  │  │ CER = edit_dist/max_len│ │    │  │                        │   │    │
│  │  │ format = regex match   │ │    │  │ IoU = area(∩)/area(∪)  │   │    │
│  │  │ anti_rep = 1-rep_rate  │ │    │  │ ID_overlap = |pred∩gt| │   │    │
│  │  └─────────────────────────┘ │    │  │           / |gt|      │   │    │
│  │                               │    │  └─────────────────────────┘   │    │
│  │  Per-group advantage:         │    │  Per-group advantage:           │    │
│  │  Aᵢ = (Rᵢ - μ₂₈) / σ₂₈     │    │  Aᵢ = (Rᵢ - μ₁₄) / σ₁₄       │    │
│  │                               │    │                                 │    │
│  │  Duration: ~3 days on 8×A100 │    │  Duration: ~2 days on 8×A100   │    │
│  └───────────────┬───────────────┘    └───────────────┬─────────────────┘    │
│                  │                                    │                      │
│                  ▼                                    ▼                      │
│  → OCR-VLM-rl (OCR accuracy: 96.2%)   → OCR-VLM-bbox-rl (mAP: 89.1%)     │
│      τ_ocr = θ_ocr_rl - θ_base            τ_bbox = θ_bbox_rl - θ_base      │
│                  │                                    │                      │
│                  └─────────────────┬──────────────────┘                      │
│                                   ▼                                          │
│  Stage 3: WEIGHT-SPACE MERGING (Model Souping)                               │
│  ══════════════════════════════════════════════                                │
│  Goal:     Combine OCR precision with bbox accuracy                          │
│  Method:   Two-phase: checkpoint averaging within each task, then             │
│            task arithmetic across tasks                                       │
│                                                                              │
│  Phase A: Intra-task soup (average last 3 checkpoints of each RL run)        │
│    θ̃_ocr = (θ_ocr_step_8K + θ_ocr_step_9K + θ_ocr_step_10K) / 3           │
│    θ̃_bbox = (θ_bbox_step_6K + θ_bbox_step_7K + θ_bbox_step_8K) / 3         │
│                                                                              │
│  Phase B: Cross-task merge (task arithmetic)                                  │
│    τ̃_ocr = θ̃_ocr - θ_base                                                  │
│    τ̃_bbox = θ̃_bbox - θ_base                                                │
│    θ_merged = θ_base + 0.8·τ̃_ocr + 0.6·τ̃_bbox                             │
│                                                                              │
│  λ tuning: grid search on 1K validation pages                                │
│                                                                              │
│  Final benchmarks:                                                           │
│  ┌────────────────────────┬──────────┬──────────┬──────────┬─────────┐      │
│  │ Model                  │ OCR CER↓ │ BBox mAP↑│ Table F1↑│ Size    │      │
│  ├────────────────────────┼──────────┼──────────┼──────────┼─────────┤      │
│  │ SFT-only (teacher KD)  │ 3.8%     │ 82.3%    │ 71.2%    │ 7B      │      │
│  │ GRPO OCR-only          │ 2.1%     │ 79.1%    │ 68.5%    │ 7B      │      │
│  │ GRPO BBox-only         │ 4.2%     │ 89.1%    │ 73.8%    │ 7B      │      │
│  │ Merged (soup)          │ 2.4%     │ 87.5%    │ 74.1%    │ 7B      │      │
│  │ Teacher (235B)         │ 1.2%     │ 93.4%    │ 82.6%    │ 235B    │      │
│  └────────────────────────┴──────────┴──────────┴──────────┴─────────┘      │
│                                                                              │
│  The merged 7B model achieves ~85% of the 235B teacher quality               │
│  at 1/30th the inference cost — and the merge was FREE (no training).       │
│                                                                              │
│  Outputs: → OCR-VLM-soup       (balanced OCR + bbox)                        │
│           → OCR-VLM-bbox-soup  (bbox-prioritized variant, λ_bbox=0.9)       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### When to Use Each Stage — Decision Guide

| Stage | Cost | Duration | What It Fixes | Skip If... |
|-------|------|----------|--------------|------------|
| **SFT** | High | Days–weeks | Foundational capabilities (OCR, QA, formatting) | Using a pretrained base that already has the skills |
| **SFT + KD** | High+ | Days–weeks | Same + transfer dark knowledge from teacher | No teacher model available |
| **RLHF/DPO** | Medium | Hours–days | Hallucinations, safety, style preferences | Task has verifiable answers (use GRPO instead) |
| **RLVR (GRPO)** | Medium | Hours–days | Task-specific precision (CER, IoU, exact match) | No auto-gradeable metric exists |
| **Weight Merging** | Near-zero | Minutes | Multi-task balance; combine specialists | Only one RL variant needed |

```
  Decision tree for choosing your pipeline:

  Do you have a strong base model (SFT already done)?
       │
       ├── NO ──▶ Stage 1: SFT (or SFT+KD if teacher available)
       │
       └── YES ──▶ What needs improving?
                       │
                       ├── Hallucinations / safety / style
                       │       │
                       │       └── Do you have preference pairs?
                       │               ├── YES ──▶ DPO (simplest)
                       │               └── NO ──▶ Collect them, then DPO
                       │
                       ├── Task-specific accuracy (OCR, math, code)
                       │       │
                       │       └── Can you auto-grade? ──▶ GRPO
                       │
                       └── Need BOTH safety + accuracy
                               │
                               └── Train DPO + GRPO separately ──▶ Merge (Stage 3)
```

---

## Techniques Cheat Sheet

| Technique | Formula / Key Idea | Memory Impact | Speed Impact |
|-----------|---------------------|---------------|--------------|
| **InfoNCE** | $\mathcal{L} = -\frac{1}{2N}\sum_i [\log P_{ii}^{\text{i→t}} + \log P_{ii}^{\text{t→i}}]$ | — | — |
| **SigLIP** | $-\frac{1}{N^2}\sum_{i,j}\log\sigma(z_{ij} s_{ij}/\tau)$ | — | Better large-batch scaling |
| **Grad Accumulation** | $\bar{g} = \frac{1}{K}\sum_k \nabla\mathcal{L}(\mathcal{B}_k) \approx \nabla\mathcal{L}(\cup_k \mathcal{B}_k)$ | 0% (activations) | $K\times$ steps per update |
| **AdamW** | $\theta \leftarrow \theta - \eta(\hat{m}/(\sqrt{\hat{v}}+\epsilon) + \lambda\theta)$ | 2× params (moments) | Baseline |
| **Cosine LR + Warmup** | $\eta \propto t/T_w$ then $\eta \propto 1 + \cos(\pi \frac{t-T_w}{T_{\text{tot}}-T_w})$ | 0% | 0% |
| **Grad Clipping** | $\hat{g} = c\, g/\|g\|$ if $\|g\| > c$ | 0% | Negligible |
| **Mixed Precision** | $\mathcal{L}_{\text{scaled}} = 2^s \mathcal{L}$; fp16/bf16 compute | ~50% | ~1.5–2× |
| **Grad Checkpointing** | Store $O(\sqrt{L})$ activations; recompute rest | ~60% activations | ~0.67× backward |
| **Freezing Encoders** | $\nabla_{\theta_{\text{enc}}} = 0$ | ~50%+ | ~2× |
| **SFT + KD** | $(1-\alpha)\text{CE} + \alpha T^2 \text{KL}(S \| T)$ | +teacher model | — |
| **DPO** | $-\log\sigma(\beta\Delta\log\pi)$; implicit reward | 2 models | — |
| **GRPO** | Group-relative advantage; verifiable rewards | 1 model + rollouts | — |
| **Task Arithmetic** | $\theta^{\text{base}} + \sum_k \lambda_k \tau_k$ | 0% (CPU merge) | 0% |
| **TIES Merging** | Trim + elect sign + disjoint merge | 0% (CPU merge) | 0% |
| **DARE + TIES** | Random dropout + TIES for interference reduction | 0% (CPU merge) | 0% |

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total notebooks | 5 |
| Total visualizations | 11 |
| Training techniques covered | 20 (InfoNCE, SigLIP, ITM, MLM, AdamW, cosine LR, grad accum, mixed precision, checkpointing, SFT, KD, RLHF/PPO, DPO, GRPO, linear soup, task arithmetic, TIES, DARE, model souping, curriculum learning) |
| RL algorithms with full derivations | 3 (RLHF/PPO, DPO, GRPO) |
| Merging methods | 4 (Linear, Task Arithmetic, TIES, DARE) |
| Full training loop built | Yes (production-quality) |
| Runs on CPU | Yes |

---

## Next Step

**[04_Finetuning_LowCompute/01_lora_from_scratch/01_lora_from_scratch.ipynb](../04_Finetuning_LowCompute/01_lora_from_scratch/01_lora_from_scratch.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/01_lora_from_scratch/01_lora_from_scratch.ipynb) — finetune with 100× fewer parameters using LoRA and QLoRA.

---

## Notebooks 4–5 (Alignment & Scaling)

| # | Notebook | Topic |
|---|----------|-------|
| 04 | [04_multimodal_alignment](04_multimodal_alignment/README.md) | Projection heads, temperature, CLIP alignment, Recall@K |
| 05 | [05_scaling_laws](05_scaling_laws/README.md) | Chinchilla scaling, compute-optimal training |
