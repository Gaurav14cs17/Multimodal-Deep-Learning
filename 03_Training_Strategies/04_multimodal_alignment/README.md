# 04 — Multimodal Alignment Theory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/04_multimodal_alignment/04_multimodal_alignment.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

Alignment is the geometric foundation of CLIP, ALIGN, and all contrastive VLMs:

- Projection heads and representation geometry on the hypersphere
- Temperature scaling — full gradient derivation
- Alignment metrics: Recall@K, R-Precision, mean rank
- Train alignment with synthetic pairs

---

## Projection Head — Derivation

### Step 1: Encoder Output

Raw encoder outputs $\mathbf{h}_I \in \mathbb{R}^{d_I}$, $\mathbf{h}_T \in \mathbb{R}^{d_T}$ may differ in dimension.

### Step 2: Linear Projection

$$
\mathbf{z}_I = W_I \mathbf{h}_I + b_I, \quad \mathbf{z}_T = W_T \mathbf{h}_T + b_T, \quad W_I \in \mathbb{R}^{d \times d_I}, W_T \in \mathbb{R}^{d \times d_T}
$$

### Step 3: L2 Normalization

$$
\hat{\mathbf{v}} = \frac{\mathbf{z}_I}{\|\mathbf{z}_I\|_2}, \quad \hat{\mathbf{t}} = \frac{\mathbf{z}_T}{\|\mathbf{z}_T\|_2}
$$

Both lie on the unit hypersphere $\mathbb{S}^{d-1}$.

**Numerical example:** $d=3$, $\mathbf{z}_I = [3, 4, 0]$:

$$
\|\mathbf{z}_I\| = 5, \quad \hat{\mathbf{v}} = [0.6, 0.8, 0.0]
$$

---

## InfoNCE Alignment Loss — Full Derivation

### Step 1: Mutual Information Lower Bound

$$
I(X; Y) \geq \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\log\frac{f(x_i, y_i)}{\frac{1}{N}\sum_{j=1}^{N}f(x_i, y_j)}\right]
$$

### Step 2: Choose Density Ratio

$$
f(x, y) = \exp(x^\top y / \tau)
$$

### Step 3: InfoNCE Loss

$$
\mathcal{L}_{\text{NCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(x_i^\top y_i / \tau)}{\sum_{j=1}^{N}\exp(x_i^\top y_j / \tau)}
$$

### Step 4: Gradient w.r.t. Temperature

Let $s_{ij} = x_i^\top y_j$ and $p_{ij} = \text{softmax}(s_{ij}/\tau)_j$:

$$
\frac{\partial \mathcal{L}}{\partial \tau} = \frac{1}{\tau^2}\sum_{i}\sum_{j}p_{ij} s_{ij} - \frac{1}{\tau^2}\sum_{i}s_{ii}
$$

**Numerical example:** $N=3$, $s_{11}=0.9, s_{12}=0.2, s_{13}=0.1$, $\tau=0.07$:

$$
p_{11} = \frac{e^{0.9/0.07}}{e^{0.9/0.07} + e^{0.2/0.07} + e^{0.1/0.07}} = \frac{e^{12.86}}{e^{12.86} + e^{2.86} + e^{1.43}} \approx 0.9999
$$

$$
\mathcal{L}_1 = -\log(0.9999) \approx 0.0001
$$

---

## Recall@K — Metric Derivation

For query image $i$, rank text $j$ by similarity $s_{ij} = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j$.

**Recall@K:** fraction of queries where correct match ranks in top $K$:

$$
\text{Recall@}K = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\text{rank}(i, i) \leq K]
$$

**Numerical example:** $N=4$, similarities for image 0 vs texts $[0.92, 0.15, 0.22, 0.18]$:

- Rank of correct text 0: 1 (highest)
- $\mathbb{1}[\text{rank} \leq 1] = 1$ → contributes to Recall@1

If similarities were $[0.15, 0.92, 0.22, 0.18]$ (wrong text highest):

- Rank of correct text 0: 3
- Recall@1 = 0, Recall@3 = 1

---

## Alignment Geometry — Hypersphere View

Matching pairs should cluster; negatives spread uniformly (under ideal alignment):

$$
\mathbb{E}_{\text{pos}}[\hat{\mathbf{v}}^\top \hat{\mathbf{t}}] \approx 1, \quad \mathbb{E}_{\text{neg}}[\hat{\mathbf{v}}^\top \hat{\mathbf{t}}] \approx 0
$$

For $d=512$, random unit vectors have expected dot product $\mathbb{E}[\cos\theta] = 0$ with variance $\approx 1/d$.

---

## Mathematical Proofs

### Proof: Projection Head Theory — Hypersphere Alignment

**Theorem:** L2-normalized projections map embeddings to $\mathbb{S}^{d-1}$, making cosine similarity the natural metric for contrastive learning.

**Step 1 — Encoder outputs** $\mathbf{h}_I \in \mathbb{R}^{d_I}$, $\mathbf{h}_T \in \mathbb{R}^{d_T}$ may differ in dimension.

**Step 2 — Linear projection to shared dim $d$:**

$$
\mathbf{z}_I = W_I \mathbf{h}_I + b_I, \quad \mathbf{z}_T = W_T \mathbf{h}_T + b_T
$$

**Why:** Projection heads decouple representation learning (encoder) from alignment learning (contrastive head) — SimCLR showed this improves downstream transfer.

**Step 3 — L2 normalization:**

$$
\hat{\mathbf{v}} = \mathbf{z}_I / \lVert \mathbf{z}_I \rVert_2, \quad \hat{\mathbf{t}} = \mathbf{z}_T / \lVert \mathbf{z}_T \rVert_2
$$

**Step 4 — Geometry:** Contrastive loss pulls $\hat{\mathbf{v}}_i$ toward $\hat{\mathbf{t}}_i$ on the sphere; negatives repel uniformly if well-trained. Expected negative similarity $\approx 0$ for high $d$. **∎**

#### Numerical Example

$\mathbf{z}_I = [3, 4, 0]$, $\lVert \mathbf{z}_I \rVert = 5$: $\hat{\mathbf{v}} = [0.6, 0.8, 0.0]$. Dot product with $\hat{\mathbf{t}} = [0.6, 0.8, 0.0]$ gives similarity $= 1.0$.

---

## What You'll Build

- Dual projection heads + L2 normalize
- InfoNCE trainer with learnable $\tau$
- Recall@1, @5, @10 evaluation on synthetic data

---

## Next Step

**[05_scaling_laws/05_scaling_laws.ipynb](../05_scaling_laws/05_scaling_laws.ipynb)**
