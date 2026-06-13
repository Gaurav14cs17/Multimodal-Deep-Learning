# 01 — CLIP from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/01_clip_from_scratch/01_clip_from_scratch.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Intermediate | **GPU Required:** No (CPU-friendly)

---

## What You'll Learn

**CLIP (Contrastive Language-Image Pre-training)** is the foundation of modern multimodal AI. This notebook builds it **piece by piece**.

- CLIP architecture — dual encoders + projection + contrastive loss
- InfoNCE loss — math, code, and step-by-step visualization
- Training on synthetic data (no downloads needed)
- Visualizing the learned embedding space
- Zero-shot classification

---

## Architecture — Full Diagram

```
  ┌───────────────────────────────────────────────────────────────┐
  │                         CLIP                                   │
  │                                                                │
  │   IMAGE BRANCH                     TEXT BRANCH                 │
  │   ────────────                     ───────────                 │
  │   Image (224×224×3)                Text ("a photo of a cat")   │
  │        │                                │                      │
  │        ▼                                ▼                      │
  │   ┌──────────┐                    ┌──────────┐                │
  │   │ ViT-B/32 │                    │ Text     │                │
  │   │ 12 layers│                    │ Transf.  │                │
  │   │ 86M      │                    │ 12 layers│                │
  │   └────┬─────┘                    └────┬─────┘                │
  │        │ v ∈ ℝ^768                     │ t ∈ ℝ^512            │
  │        ▼                                ▼                      │
  │   ┌──────────┐                    ┌──────────┐                │
  │   │ Linear   │                    │ Linear   │                │
  │   │ Proj     │                    │ Proj     │                │
  │   │ 768→512  │                    │ 512→512  │                │
  │   └────┬─────┘                    └────┬─────┘                │
  │        │ v̂ ∈ ℝ^512                    │ t̂ ∈ ℝ^512            │
  │        ▼                                ▼                      │
  │   L2-normalize                    L2-normalize                │
  │        │                                │                      │
  │        └───────────┬────────────────────┘                     │
  │                    │                                           │
  │                    ▼                                           │
  │         ┌──────────────────┐                                  │
  │         │ Cosine Similarity│   S_ij = v̂_i · t̂_j              │
  │         │ Matrix (N × N)   │                                  │
  │         └────────┬─────────┘                                  │
  │                  │                                             │
  │                  ▼                                             │
  │           InfoNCE Loss                                        │
  │    (diagonal = positive pairs)                                │
  └───────────────────────────────────────────────────────────────┘
```

---

## Key Equations

### 1. Projection + L2 Normalization

Raw encoder outputs are projected to a shared dimension and L2-normalized:

$$
\hat{\mathbf{v}}_i = \frac{W_v \cdot f_\theta(\text{img}_i)}{\lVert W_v \cdot f_\theta(\text{img}_i) \rVert}, \quad \hat{\mathbf{t}}_j = \frac{W_t \cdot g_\phi(\text{txt}_j)}{\lVert W_t \cdot g_\phi(\text{txt}_j) \rVert}
$$

After normalization, $\hat{\mathbf{v}}_i, \hat{\mathbf{t}}_j \in \mathbb{S}^{d-1}$ (the unit hypersphere).

### 2. Similarity Matrix

For a batch of $N$ pairs, compute the $N \times N$ similarity matrix:

$$
S_{ij} = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j \cdot e^{\tau}
$$

where $\tau$ is a **learnable** log-temperature (CLIP learns $\tau$ during training, initialized at $\tau = \log(1/0.07) \approx 2.66$).

### 3. Symmetric InfoNCE Loss (CLIP Loss)

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}} + \log\frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}}\right]
$$

**Breakdown of the two terms:**

| Term | Reads As | Task |
|------|----------|------|
| $-\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}$ | "Among all texts, find the match for image $i$" | Image → Text retrieval |
| $-\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}}$ | "Among all images, find the match for text $i$" | Text → Image retrieval |

### 4. Gradient Analysis — What CLIP Learns

The gradient of the loss with respect to similarity $S_{ij}$:

$$
\frac{\partial \mathcal{L}}{\partial S_{ij}} = \frac{1}{\tau}\left(P_{ij} - \mathbb{1}[i = j]\right)
$$

where $P_{ij} = \text{softmax}(S_i)_j$. **Interpretation:**
- If $i = j$ (positive pair): gradient $= \frac{1}{\tau}(P_{ii} - 1) < 0$ → **push similarity UP**
- If $i \neq j$ (negative pair): gradient $= \frac{1}{\tau} P_{ij} > 0$ → **push similarity DOWN**

### 5. Step-by-Step Numerical Example

Batch of $N = 3$ pairs, $\tau = 0.07$:

```
  Raw similarity matrix S (before /τ):
  ┌──────────────────────────────┐
  │  0.82   0.15   0.21         │  ← Image 0 vs all texts
  │  0.18   0.79   0.13         │  ← Image 1 vs all texts
  │  0.25   0.11   0.85         │  ← Image 2 vs all texts
  └──────────────────────────────┘
  Diagonal = positive pairs (should be high ✓)

  After scaling by 1/τ = 1/0.07 ≈ 14.3:
  ┌──────────────────────────────┐
  │ 11.71   2.14   3.00         │
  │  2.57  11.29   1.86         │
  │  3.57   1.57  12.14         │
  └──────────────────────────────┘

  Softmax (row-wise) → probabilities:
  ┌──────────────────────────────┐
  │  0.99   0.0001  0.0002      │  ← very confident ✓
  │  0.0002 0.99    0.0001      │
  │  0.0002 0.0000  0.99        │
  └──────────────────────────────┘

  Loss = -1/3 × [log(0.99) + log(0.99) + log(0.99)] ≈ 0.01
```

---

## Zero-Shot Classification

CLIP enables classification **without any training on labels** — just provide text descriptions:

$$
\hat{y} = \arg\max_{k} \; \hat{\mathbf{v}}^\top \hat{\mathbf{t}}_k
$$

where $\hat{\mathbf{t}}_k = \text{CLIP}_{\text{text}}(\texttt{"a photo of a [class}_k\texttt{]"})$.

```
  Test image ──► CLIP ──► v̂
                             │
                             ├── sim(v̂, "a photo of a dog")   = 0.31
                             ├── sim(v̂, "a photo of a cat")   = 0.89  ← Winner!
                             ├── sim(v̂, "a photo of a car")   = 0.12
                             └── sim(v̂, "a photo of a bird")  = 0.22
```

---

## Training Dynamics — What You'll See

```
  Similarity Matrix Evolution:

  Epoch 0 (random)        Epoch 50              Epoch 200 (converged)
  ┌───────────────┐       ┌───────────────┐     ┌───────────────┐
  │ .31 .38 .22   │       │ .65 .18 .12   │     │ .95 .02 .01   │
  │ .24 .29 .41   │  ──►  │ .11 .72 .09   │ ──► │ .01 .96 .01   │
  │ .39 .21 .33   │       │ .15 .08 .68   │     │ .01 .01 .97   │
  └───────────────┘       └───────────────┘     └───────────────┘
   Uniform noise           Diagonal emerging     Sharp diagonal
```

---

## CLIP Scaling — Original Paper Results

| Model | Image Encoder | Text Encoder | Data | ImageNet Zero-Shot |
|-------|-------------|-------------|------|--------------------|
| CLIP ViT-B/32 | ViT-B (32px patches) | 63M Transformer | 400M pairs | 63.2% |
| CLIP ViT-B/16 | ViT-B (16px patches) | 63M Transformer | 400M pairs | 68.3% |
| CLIP ViT-L/14 | ViT-L (14px patches) | 63M Transformer | 400M pairs | 75.3% |
| CLIP ViT-L/14@336 | ViT-L (14px, 336px) | 63M Transformer | 400M pairs | 76.2% |

---

## What You'll Build

- Complete CLIP model (ViT image encoder + text encoder + projection heads)
- InfoNCE contrastive training loop
- Similarity matrix visualization (watch the diagonal emerge)
- Zero-shot classification demo

---

## Prerequisites

- Module 01 notebooks (cosine similarity, encoders, fusion strategies)
- Comfortable with PyTorch `nn.Module` and training loops

---

## Next Step

**[02_image_captioning](../02_image_captioning/)** — Image → Text generation
