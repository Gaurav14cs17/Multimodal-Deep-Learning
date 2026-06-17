# 01 — What is Multimodal Learning?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb)

> **Time:** ~30 minutes | **Difficulty:** Beginner | **GPU Required:** No (CPU-friendly)

---

## What You'll Learn

This notebook answers the fundamental question: **why combine multiple modalities?**

- What "multimodal" means — images, text, audio, video working together
- The landscape of multimodal models (CLIP, LLaVA, Flamingo, GPT-4V, Gemini)
- How different modalities are represented as vectors
- The key challenge: **aligning different representation spaces**
- **Hands-on:** Build a Mini-CLIP model from scratch

---

## Why Multimodal?

Humans don't perceive the world through a single sense — we see, hear, read, and touch simultaneously. Multimodal AI mirrors this by learning **joint representations** across modalities:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     The Multimodal Landscape                     │
  │                                                                  │
  │   Modality        Representation         Dimensionality          │
  │   ─────────       ──────────────         ──────────────          │
  │   Image           Pixel grid             H × W × 3              │
  │   Text            Token IDs              Sequence of integers    │
  │   Audio           Waveform / Mel spec    1D signal / 2D spec     │
  │   Video           Frame sequence         T × H × W × 3          │
  │   3D / Point      Point cloud            N × 3                   │
  │   Tabular         Feature vector         d-dimensional           │
  │                                                                  │
  │   Challenge: Each lives in a DIFFERENT space.                    │
  │   Goal: Map them ALL into a SHARED embedding space.              │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Core Concept — Shared Embedding Space

The fundamental idea: project every modality into the **same** $d$-dimensional vector space so that semantically similar concepts (regardless of modality) land **close together**:

```
  Image                    Text
  ┌─────────┐              ┌─────────┐
  │ Pixels  │              │ Tokens  │
  │ 224×224 │              │ "a cat" │
  └────┬────┘              └────┬────┘
       │                        │
       ▼                        ▼
  ┌─────────┐              ┌─────────┐
  │ Image   │              │ Text    │
  │ Encoder │              │ Encoder │
  │ f_θ(·)  │              │ g_φ(·)  │
  └────┬────┘              └────┬────┘
       │                        │
       ▼                        ▼
    v ∈ ℝᵈ                  t ∈ ℝᵈ
       │                        │
       ▼                        ▼
  ┌──────────────────────────────────┐
  │    Shared Embedding Space (ℝᵈ)   │
  │                                  │
  │    🐱 v_cat ●──────● t_"cat"    │  ← close (positive pair)
  │                                  │
  │    🐕 v_dog ●   ● t_"sunset"   │  ← far (negative pair)
  │                                  │
  └──────────────────────────────────┘

  sim(v, t) = cos(v, t)
  "Are these about the same thing?"
```

---

## Key Equations

### 1. Cosine Similarity — Measuring Alignment

Given an image embedding $\mathbf{v} \in \mathbb{R}^d$ and a text embedding $\mathbf{t} \in \mathbb{R}^d$:

$$
\text{sim}(\mathbf{v}, \mathbf{t}) = \frac{\mathbf{v}^\top \mathbf{t}}{\lVert \mathbf{v} \rVert \cdot \lVert \mathbf{t} \rVert} \in [-1, 1]
$$

**Geometric interpretation:** cosine of the angle $\theta$ between two vectors:

$$
\cos(\theta) = \frac{\mathbf{v} \cdot \mathbf{t}}{\lVert \mathbf{v} \rVert \; \lVert \mathbf{t} \rVert}
$$

| Value | Meaning |
|-------|---------|
| $+1$ | Vectors point in the same direction — **perfect match** |
| $0$ | Vectors are orthogonal — **unrelated** |
| $-1$ | Vectors point in opposite directions — **semantic opposite** |

**Numerical example** with $d = 4$:

$$
\mathbf{v} = [0.5, 0.3, 0.8, 0.1], \quad \mathbf{t} = [0.4, 0.35, 0.75, 0.15]
$$

$$
\mathbf{v} \cdot \mathbf{t} = 0.20 + 0.105 + 0.60 + 0.015 = 0.92
$$

$$
\lVert \mathbf{v} \rVert = \sqrt{0.25 + 0.09 + 0.64 + 0.01} = \sqrt{0.99} \approx 0.995
$$

$$
\lVert \mathbf{t} \rVert = \sqrt{0.16 + 0.1225 + 0.5625 + 0.0225} = \sqrt{0.8675} \approx 0.931
$$

$$
\text{sim} = \frac{0.92}{0.995 \times 0.931} \approx 0.993 \quad \text{(very similar!)}
$$

### 2. L2 Normalization — Why We Normalize

Before computing similarity, we **L2-normalize** embeddings to the unit hypersphere:

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\lVert \mathbf{v} \rVert}, \quad \hat{\mathbf{t}} = \frac{\mathbf{t}}{\lVert \mathbf{t} \rVert}
$$

After normalization, cosine similarity becomes a simple dot product:

$$
\text{sim}(\hat{\mathbf{v}}, \hat{\mathbf{t}}) = \hat{\mathbf{v}}^\top \hat{\mathbf{t}}
$$

This is critical because it **decouples direction (semantics) from magnitude (confidence)**.

### 3. InfoNCE Loss — Contrastive Alignment

Given a batch of $N$ image-text pairs $\{(\mathbf{v}_i, \mathbf{t}_i)\}_{i=1}^N$, we want the matching pair $(i, i)$ to have high similarity while all mismatched pairs $(i, j)$ where $j \neq i$ have low similarity:

$$
\mathcal{L}_{\text{i2t}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{\text{sim}(\mathbf{v}_i, \mathbf{t}_i)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(\mathbf{v}_i, \mathbf{t}_j)/\tau}}
$$

$$
\mathcal{L}_{\text{t2i}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{\text{sim}(\mathbf{t}_i, \mathbf{v}_i)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(\mathbf{t}_i, \mathbf{v}_j)/\tau}}
$$

$$
\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2}(\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}})
$$

where $\tau > 0$ is a **temperature** parameter (typically $\tau = 0.07$).

**Intuition:** Each row/column of the $N \times N$ similarity matrix is treated as a classification problem — "which of the $N$ texts matches this image?" (and vice versa).

### 4. Temperature $\tau$ — Sharpness Control

$$
P(j \mid i) = \frac{e^{S_{ij}/\tau}}{\sum_k e^{S_{ik}/\tau}}
$$

| $\tau$ | Effect | Distribution |
|--------|--------|-------------|
| $\tau \to 0$ | Extremely sharp — approaches argmax | One-hot |
| $\tau = 0.07$ | CLIP default — confident but smooth | Peaked |
| $\tau = 1.0$ | Standard softmax | Diffuse |
| $\tau \to \infty$ | Uniform — ignores similarities | Flat |

---

## Mathematical Proofs

### Proof: Cosine Similarity — Geometric Derivation

**Theorem:** Cosine similarity equals the cosine of the angle between two vectors, and equals their dot product divided by the product of their norms.

**Step 1 — Define the angle between vectors:**

For $\mathbf{v}, \mathbf{t} \in \mathbb{R}^d \setminus \{\mathbf{0}\}$, the angle $\theta$ between them satisfies the dot-product identity:

$$
\mathbf{v}^\top \mathbf{t} = \lVert \mathbf{v} \rVert \, \lVert \mathbf{t} \rVert \cos(\theta)
$$

**Why:** This is the standard inner-product geometry in $\mathbb{R}^d$; $\cos(\theta)$ measures directional alignment independent of vector length.

**Step 2 — Solve for $\cos(\theta)$:**

Divide both sides by $\lVert \mathbf{v} \rVert \lVert \mathbf{t} \rVert$ (nonzero):

$$
\cos(\theta) = \frac{\mathbf{v}^\top \mathbf{t}}{\lVert \mathbf{v} \rVert \lVert \mathbf{t} \rVert} = \text{sim}(\mathbf{v}, \mathbf{t})
$$

**Step 3 — Geometric interpretation:**

When $\theta = 0°$, $\cos(\theta) = 1$ (same direction). When $\theta = 90°$, $\cos(\theta) = 0$ (orthogonal). When $\theta = 180°$, $\cos(\theta) = -1$ (opposite direction).

**Step 4 — Connection to L2 normalization:**

If $\hat{\mathbf{v}} = \mathbf{v}/\lVert \mathbf{v} \rVert$ and $\hat{\mathbf{t}} = \mathbf{t}/\lVert \mathbf{t} \rVert$, then $\lVert \hat{\mathbf{v}} \rVert = \lVert \hat{\mathbf{t}} \rVert = 1$, so:

$$
\text{sim}(\hat{\mathbf{v}}, \hat{\mathbf{t}}) = \hat{\mathbf{v}}^\top \hat{\mathbf{t}}
$$

**∎**

#### Numerical Example

$\mathbf{v} = [3, 0]$, $\mathbf{t} = [2, 2]$:

$$
\mathbf{v}^\top \mathbf{t} = 6, \quad \lVert \mathbf{v} \rVert = 3, \quad \lVert \mathbf{t} \rVert = \sqrt{8} \approx 2.828
$$

$$
\text{sim} = \frac{6}{3 \times 2.828} = \frac{6}{8.485} \approx 0.707 = \cos(45°)
$$

---

### Proof: InfoNCE Loss from Mutual Information

**Theorem:** The InfoNCE loss provides a lower bound on mutual information $I(X; Y)$.

**Step 1 — Define mutual information:**

$$
I(X; Y) = \mathbb{E}_{p(x,y)}\left[\log \frac{p(x \mid y)}{p(x)}\right]
$$

**Why:** MI measures how much knowing $Y$ reduces uncertainty about $X$; the log density ratio is the sufficient statistic.

**Step 2 — Introduce a critic function $f(x, y)$ as a density ratio estimator:**

We don't know $p(x \mid y)/p(x)$ exactly, so we approximate it with a learned function:

$$
\frac{p(x \mid y)}{p(x)} \approx \frac{e^{f(x, y)}}{\frac{1}{N}\sum_{j=1}^{N} e^{f(x_j, y)}}
$$

**Why:** The denominator normalizes over $N$ samples drawn from $p(x)$, turning the ratio into a valid classifier score.

**Step 3 — Substitute into MI and apply Jensen's inequality:**

Since $\log$ is concave, $\mathbb{E}[\log X] \leq \log \mathbb{E}[X]$, which rearranges to a lower bound:

$$
I(X; Y) \geq \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{f(x_i, y_i)}}{\frac{1}{N}\sum_{j=1}^{N}e^{f(x_j, y_i)}}\right]
$$

**Step 4 — Choose $f(x, y) = \text{sim}(x, y) / \tau$ (cosine similarity scaled by temperature):**

$$
\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{\text{sim}(x_i, y_i)/\tau}}{\sum_{j=1}^{N}e^{\text{sim}(x_i, y_j)/\tau}}
$$

**Step 5 — Bound tightness:**

The bound becomes tight as $N \to \infty$:

$$
I(X; Y) \geq \log(N) - \mathcal{L}_{\text{InfoNCE}}
$$

**∎**

#### Numerical Example

Given $N=3$ samples with cosine similarities and $\tau = 0.5$:

| Pair | $\text{sim}(x_i, y_j)$ |
|------|------------------------|
| (1,1) | 0.9 |
| (1,2) | 0.2 |
| (1,3) | 0.1 |

$$
\mathcal{L}_1 = -\log\frac{e^{0.9/0.5}}{e^{0.9/0.5} + e^{0.2/0.5} + e^{0.1/0.5}} = -\log\frac{e^{1.8}}{e^{1.8} + e^{0.4} + e^{0.2}}
$$

$$
= -\log\frac{6.050}{6.050 + 1.492 + 1.221} = -\log\frac{6.050}{8.763} = -\log(0.690) = 0.371
$$

---

## Multimodal Model Landscape

| Model | Year | Modalities | Key Innovation |
|-------|------|-----------|---------------|
| CLIP | 2021 | Image + Text | Contrastive pre-training on 400M pairs |
| ALIGN | 2021 | Image + Text | Noisy 1.8B pairs (no manual curation) |
| Flamingo | 2022 | Image + Text | Few-shot visual reasoning via gated xattn |
| BLIP-2 | 2023 | Image + Text | Q-Former bridge (frozen ViT + frozen LLM) |
| LLaVA | 2023 | Image + Text | Simple linear projector + instruction tuning |
| GPT-4V | 2023 | Image + Text | Proprietary, state-of-the-art |
| Gemini | 2024 | Image + Text + Audio + Video | Natively multimodal from pre-training |
| ImageBind | 2023 | 6 modalities | One shared space for image, text, audio, depth, thermal, IMU |

---

## What You'll Build

A **Mini-CLIP** model with:

```
  ┌──────────────────────────────────────────────────────┐
  │                    Mini-CLIP                          │
  │                                                      │
  │   Image Branch:                                      │
  │   ┌──────────┐     ┌────────┐     ┌──────┐          │
  │   │ CNN      │ ──► │ Flatten │ ──► │ Proj │ ──► v̂   │
  │   │ (3 conv) │     │        │     │ (d)  │          │
  │   └──────────┘     └────────┘     └──────┘          │
  │                                                      │
  │   Text Branch:                                       │
  │   ┌──────────┐     ┌────────┐     ┌──────┐          │
  │   │ Embedding│ ──► │ Pool   │ ──► │ Proj │ ──► t̂   │
  │   │ (vocab)  │     │ (mean) │     │ (d)  │          │
  │   └──────────┘     └────────┘     └──────┘          │
  │                                                      │
  │   Loss: InfoNCE(v̂, t̂)                               │
  └──────────────────────────────────────────────────────┘
```

- Simple CNN image encoder (3 convolutional layers)
- Embedding-based text encoder with mean pooling
- Projection heads mapping both to shared $\mathbb{R}^d$
- Contrastive loss training loop
- Visualization of the learned embedding space

---

## Prerequisites

- Basic PyTorch (tensors, `nn.Module`, training loops)
- Understanding of embeddings and similarity
- Linear algebra basics (dot product, norms)

---

## 🔬 Worked Examples in the Notebook

The notebook includes **5 hands-on examples** with full numerical traces:

### Example 1: Train Mini-CLIP from Scratch
Train a contrastive model on 200 synthetic pairs and watch alignment emerge in real-time. Track loss, temperature, and diagonal similarity.

### Example 2: Embedding Space Visualization
After training, project embeddings to 2D via PCA. See matching image-text pairs cluster together — visual proof that contrastive learning works.

### Example 3: InfoNCE Loss — Complete Numerical Walkthrough
Trace InfoNCE step-by-step with a batch of 4:
- Cosine similarity matrix → scale by τ=0.07 → softmax → cross-entropy
- Compare: aligned model (loss=0.0004) vs confused model (loss≈1.386)
- Verify every number with PyTorch code

### Example 4: Zero-Shot Classification
Classify synthetic images into 5 categories using ONLY text prompts — no task-specific training. Includes confusion matrix and per-class accuracy breakdown.

### Example 5: Real-World Applications Table
Industry applications: medical image search, autonomous driving, e-commerce, content moderation, accessibility, document understanding.

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb)

---

## 📄 Paper Figures in the Notebook

The notebook includes recreated architecture diagrams from key papers:

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| CLIP Overview (`../../assets/paper_figures/clip_overview.png`) | Radford et al. — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | 2021 | Official contrastive pretraining diagram from OpenAI |
| CLIP Contrastive Pretraining + Zero-Shot | Radford et al. — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | 2021 | Image-text alignment via contrastive learning |
| Flamingo Gated Cross-Attention | Alayrac et al. — [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) | 2022 | Few-shot multimodal learning, frozen LM + trainable x-attn |
| DALL·E / Stable Diffusion Pipeline | Ramesh et al. / Rombach et al. — [arXiv:2102.12092](https://arxiv.org/abs/2102.12092) | 2021-22 | Text-to-image generation via diffusion |

### Additional Papers Covered

- **CLAP** (Elizalde et al., 2023) — Audio-text contrastive learning, extends CLIP to audio
- **ALIGN** (Jia et al., 2021) — 1.8B noisy image-text pairs, scale > curation
- **ImageBind** (Girdhar et al., 2023) — 6 modalities bound through images
- **Gemini** (Google, 2023) — Natively multimodal from pretraining

---

## Next Step

**[02_modality_encoders](../02_modality_encoders/)** — How ViT and BERT encode images and text
