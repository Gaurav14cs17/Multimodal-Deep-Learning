# Module 03: Training Strategies

> **Time:** 2–3 hours | **Notebooks:** 3 | **Visuals:** 11 plots | **Difficulty:** Intermediate–Advanced
>
> **This is a CORE FOCUS module** — mastering training is what separates "I understand the theory" from "I can actually build this."

| Notebook | Open in Colab |
|----------|---------------|
| `01_contrastive_learning.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning.ipynb) |
| `02_pretraining_objectives.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives.ipynb) |
| `03_training_pipeline.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline.ipynb) |

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

## Notebook 1: `01_contrastive_learning.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning.ipynb)

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

$$S = \begin{pmatrix} 0.5 & 0.1 & 0.3 & 0.2 \\ 0.2 & 0.6 & 0.1 & 0.3 \\ 0.3 & 0.2 & 0.4 & 0.1 \\ 0.1 & 0.3 & 0.2 & 0.5 \end{pmatrix}$$

**Step 1 — Scale by temperature** $\tau = 0.07$:

$$\frac{S}{\tau} = \begin{pmatrix} 7.143 & 1.429 & 4.286 & 2.857 \\ 2.857 & 8.571 & 1.429 & 4.286 \\ 4.286 & 2.857 & 5.714 & 1.429 \\ 1.429 & 4.286 & 2.857 & 7.143 \end{pmatrix}$$

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

## Notebook 2: `02_pretraining_objectives.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives.ipynb)

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

$$\mathcal{L}_{\text{Gen}} = -\sum_{t=1}^{T} \log P(w_t \mid w_{<t},\, I)$$

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

## Notebook 3: `03_training_pipeline.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline.ipynb)

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

$$\eta(t) = \begin{cases}
\displaystyle \eta_{\max} \cdot \frac{t}{T_w} & \text{if } t < T_w \\[10pt]
\displaystyle \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\pi \cdot \frac{t - T_w}{T_{\text{tot}} - T_w}\right)\right) & \text{if } T_w \leq t \leq T_{\text{tot}}
\end{cases}$$

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

$$\hat{g} = \begin{cases}
g & \text{if } \|g\|_2 \leq c \\
\displaystyle c \cdot \frac{g}{\|g\|_2} & \text{if } \|g\|_2 > c
\end{cases}$$

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

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total notebooks | 3 |
| Total visualizations | 11 |
| Training techniques covered | 9 (InfoNCE, SigLIP, ITM, MLM, AdamW, cosine LR, grad accum, mixed precision, checkpointing) |
| Full training loop built | Yes (production-quality) |
| Runs on CPU | Yes |

---

## Next Step

**[04_Finetuning_LowCompute/01_lora_from_scratch.ipynb](../04_Finetuning_LowCompute/01_lora_from_scratch.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/01_lora_from_scratch.ipynb) — finetune with 100× fewer parameters using LoRA and QLoRA.
