# Module 01: Multimodal Foundations

> **Time:** 2-3 hours | **Notebooks:** 6 | **Visuals:** 12+ plots | **Difficulty:** Intermediate

| Notebook | Open in Colab |
|----------|---------------|
| `01_what_is_multimodal/01_what_is_multimodal.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb) |
| `02_modality_encoders/02_modality_encoders.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/02_modality_encoders/02_modality_encoders.ipynb) |
| `03_fusion_strategies/03_fusion_strategies.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/03_fusion_strategies/03_fusion_strategies.ipynb) |
| `04_attention_mechanism/04_attention_mechanism.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/04_attention_mechanism/04_attention_mechanism.ipynb) |
| `05_transformer_architecture/05_transformer_architecture.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/05_transformer_architecture/05_transformer_architecture.ipynb) |
| `06_tokenization_embeddings/06_tokenization_embeddings.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/06_tokenization_embeddings/06_tokenization_embeddings.ipynb) |

---

## The Big Idea

You already understand transformers for a single modality (text **or** images). Multimodal deep learning asks a harder question: how do you combine **images + text + audio** into a single model that reasons across all of them?

The answer rests on three pillars:

1. **Separate encoders** — each modality has its own inductive bias (convolution for images, tokenization for text).
2. **Shared embedding space** — every encoder maps to vectors in $\mathbb{R}^D$.
3. **Alignment training** — matching pairs are pulled together; non-matching pairs are pushed apart.

```
             +--------+     +--------+     +--------+
             | Image  |     | Text   |     | Audio  |
             | Input  |     | Input  |     | Input  |
             +---+----+     +---+----+     +---+----+
                 |              |              |
                 v              v              v
           +---------+   +---------+   +---------+
           | Image   |   | Text    |   | Audio   |
           | Encoder |   | Encoder |   | Encoder |
           +---------+   +---------+   +---------+
                 |              |              |
                 v              v              v
            [D-dim]         [D-dim]         [D-dim]
                 |              |              |
                 +-------+------+------+-------+
                         |             |
                         v             v
                    +---------+   +----------+
                    |  FUSION |   | ALIGNMENT|
                    +---------+   +----------+
                         |
                         v
                    +----------+
                    |  OUTPUT  |
                    +----------+
```

Every modality gets its own encoder, but they all produce vectors in the **same shared space**. That geometric agreement is what makes cross-modal understanding possible.

---

## Notebook 1: `01_what_is_multimodal/01_what_is_multimodal.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb)

### The Multimodal Landscape — A Timeline

Multimodal AI did not arrive overnight. Each milestone solved a specific sub-problem: vision-only classification, vision–language alignment, generative captioning, LLM-grounded vision, and finally omni-modal models.

```
 2020          2021          2022          2023          2024
  |             |             |             |             |
  ViT          CLIP          BLIP         LLaVA         GPT-4V
  |            ALIGN         BLIP-2       Gemini        LLaVA-1.5
  |             |           Flamingo     ImageBind     Qwen-VL
  |             |             |             |             |
  v             v             v             v             v
[Vision]     [V+L Align]   [V+L Gen]    [V+L+LLM]      [Omni]
 only        contrastive   captioning   instruction    unified
             learning      + VQA        tuning         perception
```

**Reading the timeline:**

| Era | Representative models | Core innovation |
|-----|----------------------|-----------------|
| 2020 | ViT | Treat images as token sequences; pure transformer for vision |
| 2021 | CLIP, ALIGN | Contrastive alignment of image and text in shared space |
| 2022 | BLIP, Flamingo | Generative + cross-attention fusion with frozen LLMs |
| 2023 | LLaVA, Gemini | Instruction-tuned multimodal LLMs |
| 2024 | GPT-4V, Qwen-VL | Production-scale omni-modal reasoning |

The through-line: **encode each modality → place in shared space → fuse or align**.

---

### Cosine Similarity — Full Derivation

Cosine similarity is the workhorse metric for multimodal alignment. We derive it from first principles.

#### Step 1: Dot Product

Given two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$, the **dot product** (inner product) is:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos\theta
$$

where $\theta$ is the angle between $\mathbf{a}$ and $\mathbf{b}$ in $\mathbb{R}^d$.

#### Step 2: Geometric Meaning

Rearranging:

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

This is **cosine similarity**:

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \, \|\mathbf{b}\|_2} = \cos\theta
$$

Geometrically, cosine similarity measures **directional agreement**, not magnitude. Two vectors pointing the same way score $+1$ regardless of length; opposite directions score $-1$; orthogonal vectors score $0$.

```
        b
       /
      /  θ  ← angle between a and b
     /___
    a

  cos(θ) = 1   → same direction      (most similar)
  cos(θ) = 0   → orthogonal          (unrelated)
  cos(θ) = -1  → opposite direction  (most dissimilar)
```

#### Step 3: Range and Properties

**Range:** Since $\theta \in [0, \pi]$, we have $\cos\theta \in [-1, 1]$.

**Properties:**

| Property | Formula / statement |
|----------|---------------------|
| Symmetry | $\text{sim}(\mathbf{a}, \mathbf{b}) = \text{sim}(\mathbf{b}, \mathbf{a})$ |
| Scale invariance | $\text{sim}(c\mathbf{a}, \mathbf{b}) = \text{sim}(\mathbf{a}, \mathbf{b})$ for $c > 0$ |
| Self-similarity | $\text{sim}(\mathbf{a}, \mathbf{a}) = 1$ |
| Cauchy–Schwarz bound | $|\mathbf{a} \cdot \mathbf{b}| \leq \|\mathbf{a}\| \|\mathbf{b}\|$ ensures range |

**Why cosine and not Euclidean distance?** In high-dimensional embedding spaces, vector **norm** can vary wildly across samples. Cosine similarity ignores norm and compares only direction — exactly what we want when asking "do this image and this caption describe the same thing?"

---

### The Alignment Problem — Mathematical Formulation

Given an image $I$ and text $T$, define encoders $f_I: \mathcal{I} \to \mathbb{R}^D$ and $f_T: \mathcal{T} \to \mathbb{R}^D$. The **alignment score** is:

$$
s(I, T) = \text{sim}\bigl(f_I(I), \, f_T(T)\bigr) = \frac{f_I(I)^\top f_T(T)}{\|f_I(I)\|_2 \, \|f_T(T)\|_2}
$$

**Before training** (random encoders), matching and non-matching pairs are indistinguishable:

$$
s(I_{\text{cat}}, T_{\text{"a cat"}}) \approx s(I_{\text{cat}}, T_{\text{"a dog"}}) \approx 0
$$

**After training** (aligned encoders), matching pairs dominate:

$$
s(I_{\text{cat}}, T_{\text{"a cat"}}) \gg s(I_{\text{cat}}, T_{\text{"a dog"}})
$$

Formally, for a batch of $B$ image–text pairs $\{(I_i, T_i)\}_{i=1}^B$, we want:

$$
s(I_i, T_i) > s(I_i, T_j) \quad \forall j \neq i
$$

```
BEFORE Training:                    AFTER Training:
  *                                   * .
     *  .                            *.
  .       *                         *.
    .  *                             *.
  *    .                            * .

  * = image embeddings               Matching pairs
  . = text embeddings                cluster together!
```

This is a **metric learning** problem: learn encoders such that semantically related pairs are neighbors on the unit hypersphere $\mathbb{S}^{D-1}$.

---

### Contrastive Objective — Preview

The standard way to enforce alignment is **InfoNCE** (used in CLIP). For a batch of $B$ pairs, define similarity matrix $S \in \mathbb{R}^{B \times B}$ with $S_{ij} = s(I_i, T_j) / \tau$, where $\tau > 0$ is a **temperature** hyperparameter.

The image-to-text contrastive loss for image $i$ is:

$$
\mathcal{L}_{i \to t} = -\log \frac{\exp(S_{ii})}{\sum_{j=1}^{B} \exp(S_{ij})}
$$

Symmetrically, the text-to-image loss:

$$
\mathcal{L}_{t \to i} = -\log \frac{\exp(S_{ii})}{\sum_{j=1}^{B} \exp(S_{ji})}
$$

Total loss:

$$
\mathcal{L}_{\text{contrastive}} = \frac{1}{2B} \sum_{i=1}^{B} \left( \mathcal{L}_{i \to t} + \mathcal{L}_{t \to i} \right)
$$

**Intuition:** For each image, the model must pick the correct caption from $B$ distractors. Temperature $\tau$ controls sharpness — smaller $\tau$ makes the softmax more peaked, forcing harder discrimination.

We build a **Mini-CLIP** in Notebook 1 with exactly this structure:

```python
class ImageEncoder(nn.Module):    # CNN/ViT -> D-dim vector
class TextEncoder(nn.Module):     # Embedding + Transformer -> D-dim
class MiniCLIP(nn.Module):        # Cosine similarity + temperature τ
```

### Key Takeaway (Notebook 1)

> Multimodal = **Separate Encoders** + **Shared Embedding Space** + **Contrastive Alignment Training**

---

## Notebook 2: `02_modality_encoders/02_modality_encoders.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/02_modality_encoders/02_modality_encoders.ipynb)

This is the deepest section of the module. We build a Vision Transformer (ViT) and Text Encoder from scratch, deriving every operation with full tensor shapes.

---

### 1. Patch Extraction

Given an image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, we partition it into non-overlapping patches of size $P \times P$.

**Reshaping operations:**

$$
\mathbf{x} \in \mathbb{R}^{H \times W \times C} \xrightarrow{\text{reshape}} \mathbf{x}' \in \mathbb{R}^{\frac{H}{P} \times P \times \frac{W}{P} \times P \times C}
$$

$$
\xrightarrow{\text{permute & flatten}} \mathbf{X}_p \in \mathbb{R}^{N \times (P^2 C)}
$$

where the number of patches is:

$$
N = \frac{H}{P} \cdot \frac{W}{P} = \frac{H \times W}{P^2}
$$

Each patch is a flattened vector:

$$
\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}, \quad i = 1, 2, \ldots, N
$$

**Concrete example** ($H = W = 32$, $P = 4$, $C = 3$):

$$
N = \frac{32 \times 32}{4^2} = \frac{1024}{16} = 64 \text{ patches}, \quad \text{each } \mathbf{x}_p^i \in \mathbb{R}^{48}
$$

```
Input image (32×32×3):

+----+----+----+----+----+----+----+----+
| p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |   Row 1 of patch grid (8 patches)
+----+----+----+----+----+----+----+----+
| p9 |...                                          ...
+----+ ...
...
+----+----+----+----+----+----+----+----+
|... | p64|                                         Last patch
+----+----+----+----+----+----+----+----+

Each patch p_i: reshape 4×4×3 → R^{48}
Total: N = 64 patches
```

---

### 2. Linear Projection

Each patch $\mathbf{x}_p^i \in \mathbb{R}^{P^2 C}$ is projected to model dimension $D$ via a learnable matrix $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$:

$$
\mathbf{z}_0^i = \mathbf{x}_p^i \, \mathbf{E} + \mathbf{e}_{\text{pos}}^i
$$

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{x}_p^i$ | $(P^2 C,)$ | Flattened patch |
| $\mathbf{E}$ | $(P^2 C, D)$ | Patch projection weights |
| $\mathbf{e}_{\text{pos}}^i$ | $(D,)$ | Positional encoding for patch $i$ |
| $\mathbf{z}_0^i$ | $(D,)$ | Patch token embedding |

Stacking all patches: $\mathbf{Z}_0^{\text{patches}} \in \mathbb{R}^{N \times D}$.

We prepend a learnable **[CLS]** token $\mathbf{z}_{\text{cls}} \in \mathbb{R}^D$:

$$
\mathbf{Z}_0 = \bigl[\mathbf{z}_{\text{cls}}; \; \mathbf{z}_0^1; \; \mathbf{z}_0^2; \; \ldots; \; \mathbf{z}_0^N\bigr] + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}
$$

where $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ is the full positional encoding matrix (learned or sinusoidal).

---

### 3. Positional Encoding

Transformers are permutation-invariant without position information. Positional encodings inject **where** each token sits in the sequence.

#### Sinusoidal Encoding (Vaswani et al., 2017)

For position $\text{pos}$ and dimension index $i$:

$$
PE(\text{pos},\, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

$$
PE(\text{pos},\, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

where $d$ is the model dimension $D$. Even dimensions get sine; odd dimensions get cosine. Wavelengths range from $2\pi$ (high-frequency, fine position) to $10000 \cdot 2\pi$ (low-frequency, coarse position).

#### Why Sinusoidal Works — Relative Position via Rotation

**Claim:** There exists a linear transformation $\mathbf{M}_{\Delta}$ such that:

$$
PE(\text{pos} + \Delta) = \mathbf{M}_{\Delta} \, PE(\text{pos})
$$

**Proof sketch:** Consider the 2-dimensional subspace spanned by dimensions $(2i, 2i+1)$. Define:

$$
\omega_i = \frac{1}{10000^{2i/d}}
$$

Then:

$$
\begin{pmatrix} PE(\text{pos}, 2i) \\ PE(\text{pos}, 2i+1) \end{pmatrix} = \begin{pmatrix} \sin(\omega_i \cdot \text{pos}) \\ \cos(\omega_i \cdot \text{pos}) \end{pmatrix}
$$

This is a point on the unit circle at angle $\omega_i \cdot \text{pos}$. Adding $\Delta$ to the position rotates by $\omega_i \cdot \Delta$:

$$
\begin{pmatrix} \sin(\omega_i(\text{pos}+\Delta)) \\ \cos(\omega_i(\text{pos}+\Delta)) \end{pmatrix} = \underbrace{\begin{pmatrix} \cos(\omega_i \Delta) & \sin(\omega_i \Delta) \\ -\sin(\omega_i \Delta) & \cos(\omega_i \Delta) \end{pmatrix}}_{\text{Rotation matrix } \mathbf{R}(\omega_i \Delta)} \begin{pmatrix} \sin(\omega_i \cdot \text{pos}) \\ \cos(\omega_i \cdot \text{pos}) \end{pmatrix}
$$

So relative position $\Delta$ corresponds to a **fixed rotation** — the model can learn to attend based on relative offsets.

#### Learned vs Sinusoidal

| Property | Sinusoidal | Learned |
|----------|-----------|---------|
| Parameters | 0 (fixed formula) | $(N_{\max}+1) \times D$ |
| Extrapolation beyond train length | Good (by design) | Poor (unseen positions) |
| Relative position structure | Built-in (rotation) | Must be learned |
| Used in | Original Transformer, BERT (optional) | ViT, BERT (default), our Mini-ViT |
| Best when | Variable sequence lengths | Fixed or bounded length |

ViT typically uses **learned** positional embeddings because image patch count is fixed at train time.

---

### 4. Self-Attention — Full Derivation

Self-attention lets each token query all other tokens and aggregate information weighted by relevance.

#### Query, Key, Value Projections

Given input $\mathbf{Z} \in \mathbb{R}^{(N+1) \times D}$:

$$
\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}_V
$$

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{Z}$ | $((N+1), D)$ | Input token embeddings |
| $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ | $(D, d_k)$ | Projection matrices |
| $\mathbf{Q}, \mathbf{K}$ | $((N+1), d_k)$ | Queries and keys |
| $\mathbf{V}$ | $((N+1), d_v)$ | Values (typically $d_v = d_k$) |

#### Scaled Dot-Product Attention

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

The attention weight matrix $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}) \in \mathbb{R}^{(N+1) \times (N+1)}$ satisfies $\sum_j A_{ij} = 1$ for each row $i$.

#### Why Divide by $\sqrt{d_k}$? — Variance Analysis

Let $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ be a single query and key vector. Suppose each component $q_\ell, k_\ell$ is independent with:

$$
\mathbb{E}[q_\ell] = \mathbb{E}[k_\ell] = 0, \quad \text{Var}(q_\ell) = \text{Var}(k_\ell) = 1
$$

The dot product is $s = \mathbf{q} \cdot \mathbf{k} = \sum_{\ell=1}^{d_k} q_\ell k_\ell$.

Since $q_\ell, k_\ell$ are independent with zero mean and unit variance:

$$
\mathbb{E}[q_\ell k_\ell] = \mathbb{E}[q_\ell]\mathbb{E}[k_\ell] = 0
$$

$$
\text{Var}(q_\ell k_\ell) = \mathbb{E}[q_\ell^2 k_\ell^2] - (\mathbb{E}[q_\ell k_\ell])^2 = \mathbb{E}[q_\ell^2]\mathbb{E}[k_\ell^2] = 1 \cdot 1 = 1
$$

For independent summands:

$$
\text{Var}(\mathbf{q} \cdot \mathbf{k}) = \sum_{\ell=1}^{d_k} \text{Var}(q_\ell k_\ell) = d_k
$$

As $d_k$ grows, dot products grow in magnitude → softmax inputs become very large → **softmax saturation** (gradients vanish, attention becomes nearly one-hot).

Dividing by $\sqrt{d_k}$:

$$
\text{Var}\!\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

This keeps softmax inputs at unit variance regardless of $d_k$.

#### Softmax Saturation

```
Without scaling (d_k = 64):
  logits = [12.3, 11.8, 2.1, 1.9, ...]  → softmax ≈ [0.62, 0.38, ~0, ~0, ...]
  (nearly one-hot → tiny gradients for non-max positions)

With scaling (/ √64 = 8):
  logits = [1.54, 1.48, 0.26, 0.24, ...]  → softmax ≈ [0.28, 0.26, 0.15, 0.14, ...]
  (softer distribution → healthy gradients)
```

#### Worked Numerical Example — 3 Tokens, $d_k = 4$

Suppose (after projection):

$$
\mathbf{Q} = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{pmatrix}, \quad
\mathbf{K} = \begin{pmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 \end{pmatrix}, \quad
\mathbf{V} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}
$$

**Step 1:** Compute $\mathbf{Q}\mathbf{K}^\top$:

$$
\mathbf{Q}\mathbf{K}^\top = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix}
$$

**Step 2:** Scale by $\sqrt{d_k} = \sqrt{4} = 2$:

$$
\frac{\mathbf{Q}\mathbf{K}^\top}{2} = \begin{pmatrix} 0.5 & 0 & 1.0 \\ 0 & 0.5 & 0 \\ 0.5 & 0 & 0.5 \end{pmatrix}
$$

**Step 3:** Row-wise softmax (token 0 attends to keys 0, 1, 2):

$$
A_{0,:} = \text{softmax}([0.5,\, 0,\, 1.0]) = \left[\frac{e^{0.5}}{e^{0.5}+1+e^{1.0}},\; \frac{1}{e^{0.5}+1+e^{1.0}},\; \frac{e^{1.0}}{e^{0.5}+1+e^{1.0}}\right] \approx [0.32,\; 0.19,\; 0.49]
$$

Similarly: $A_{1,:} \approx [0.25,\; 0.50,\; 0.25]$, $A_{2,:} \approx [0.38,\; 0.23,\; 0.38]$.

**Step 4:** Output = $\mathbf{A}\mathbf{V}$:

$$
\text{Output} = \begin{pmatrix} 0.32 & 0.19 \\ 0.25 & 0.50 \\ 0.38 & 0.23 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix} \approx \begin{pmatrix} 0.51 & 0.68 \\ 0.75 & 0.75 \\ 0.61 & 0.61 \end{pmatrix}
$$

Each output row is a convex combination of value vectors, weighted by attention.

---

### 5. Multi-Head Attention

Instead of one attention function, we run $h$ parallel **heads**, each attending to different subspaces:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^i,\; \mathbf{K}\mathbf{W}_K^i,\; \mathbf{V}\mathbf{W}_V^i)
$$

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, \mathbf{W}^O
$$

| Tensor | Shape |
|--------|-------|
| $\mathbf{W}_Q^i, \mathbf{W}_K^i, \mathbf{W}_V^i$ | $(D, d_k)$ where $d_k = D/h$ |
| $\mathbf{W}^O$ | $(h \cdot d_v, D)$ where $d_v = D/h$ |

#### Dimension Analysis

If $d_{\text{model}} = D = 256$ and $h = 4$ heads:

$$
d_k = d_v = \frac{D}{h} = \frac{256}{4} = 64
$$

Each head operates in a 64-dimensional subspace; concatenating 4 heads recovers $4 \times 64 = 256 = D$.

#### Parameter Count

Per head: $3 \times D \times d_k$ (for $W_Q, W_K, W_V$). Across $h$ heads: $3h \times D \times d_k = 3D^2$ (since $h \cdot d_k = D$).

Output projection: $D \times D$.

**Total:** $3D^2 + D^2 = 4D^2$.

For $D = 256$: $4 \times 256^2 = 262{,}144$ parameters in one multi-head attention layer.

#### Why Multiple Heads?

Each head can specialize in different relational patterns:

```
Head 1: local / positional patterns    (nearby patches attend to each other)
Head 2: semantic / object parts        (cat ear patch ↔ cat face patch)
Head 3: global context                 ([CLS] token aggregates everything)
Head 4: cross-region dependencies      (left object ↔ right object)
```

Single-head attention must compress all these into one $d_k$-dimensional subspace; multi-head provides **parallel representation subspaces**.

---

### 6. Layer Normalization

Layer normalization stabilizes activations by normalizing across the feature dimension:

$$
\text{LN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}
$$

where for $\mathbf{x} \in \mathbb{R}^D$:

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i, \qquad \sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2
$$

$\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^D$ are learnable scale and shift; $\epsilon \approx 10^{-5}$ prevents division by zero.

#### Pre-Norm vs Post-Norm

**Post-Norm** (original Transformer):

$$
\mathbf{Z}' = \text{LN}(\text{MSA}(\mathbf{Z}) + \mathbf{Z}), \qquad \mathbf{Z}'' = \text{LN}(\text{FFN}(\mathbf{Z}') + \mathbf{Z}')
$$

**Pre-Norm** (modern default, used in ViT):

$$
\mathbf{Z}' = \text{MSA}(\text{LN}(\mathbf{Z})) + \mathbf{Z}, \qquad \mathbf{Z}'' = \text{FFN}(\text{LN}(\mathbf{Z}')) + \mathbf{Z}'
$$

```
Post-Norm:                          Pre-Norm:
  Z ──→ MSA ──→ Add ──→ LN          Z ──→ LN ──→ MSA ──→ Add
              ↑                                    ↑
              └──────── Z                          └──── Z

  (normalize AFTER residual)          (normalize BEFORE sublayer)
  Harder to train deep models         Better gradient flow
```

Pre-norm places normalization **inside** the residual branch, keeping the residual path clean. This is why ViT and most modern transformers use pre-norm.

---

### 7. Feed-Forward Network (FFN)

Each transformer block contains a position-wise FFN applied independently to each token:

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

| Tensor | Shape |
|--------|-------|
| $\mathbf{W}_1$ | $(D, 4D)$ |
| $\mathbf{W}_2$ | $(4D, D)$ |
| $\mathbf{b}_1$ | $(4D,)$ |
| $\mathbf{b}_2$ | $(D,)$ |

#### GELU Activation

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715 x^3)\right]\right)
$$

where $\Phi(x)$ is the standard Gaussian CDF. GELU is smooth (unlike ReLU's kink at 0) and stochastically motivated — it gates the input by its probability of being positive.

#### Why 4× Expansion?

The FFN is the **primary capacity** of each transformer block. Attention mixes information between tokens; FFN transforms each token's representation individually. The $4D$ hidden dimension (standard since BERT/ViT) balances expressiveness and compute.

**Parameter count:** $D \cdot 4D + 4D \cdot D = 8D^2$ (plus biases: $5D$).

For $D = 128$: $8 \times 128^2 = 131{,}072$ FFN parameters per block.

---

### 8. Full ViT Pipeline — Tensor Shapes at Every Step

Complete forward pass for our Mini-ViT ($H=W=32$, $P=4$, $C=3$, $D=128$, $L=4$ layers, $h=4$ heads):

```
Step                          Tensor Shape              Operation
─────────────────────────────────────────────────────────────────────────
Input image                   (B, 3, 32, 32)            Raw RGB
Patch extraction              (B, 64, 48)               Reshape + flatten patches
Linear projection E           (B, 64, 128)              x_p @ E
Prepend [CLS]                 (B, 65, 128)              cat([CLS], patches)
Add positional encoding       (B, 65, 128)              + E_pos
─────────────────────────────────────────────────────────────────────────
For each layer l = 1..4:
  LayerNorm                   (B, 65, 128)              LN(Z)
  Multi-Head Self-Attention   (B, 65, 128)              4 heads, d_k=32
  Residual add                (B, 65, 128)              + Z
  LayerNorm                   (B, 65, 128)              LN(Z')
  FFN (128→512→128)           (B, 65, 128)              W1: (128,512), W2: (512,128)
  Residual add                (B, 65, 128)              + Z'
─────────────────────────────────────────────────────────────────────────
Extract [CLS] token           (B, 128)                  Z[:, 0, :]
Final LayerNorm               (B, 128)                  h_img = LN([CLS])
```

**ASCII pipeline:**

```
(B,3,32,32) ──→ patches ──→ (B,64,48) ──→ proj ──→ (B,64,128)
                                              │
                                         + [CLS]
                                              │
                                         (B,65,128) ──→ + pos ──→ Transformer×4 ──→ [CLS] ──→ (B,128)
                                                                                              = h_img
```

---

### 9. Text Encoder

The text encoder mirrors ViT's transformer stack but replaces patch extraction with **token embedding lookup**.

#### Embedding Lookup

Given token IDs $[w_1, w_2, \ldots, w_T]$ where $w_t \in \{0, 1, \ldots, V-1\}$:

$$
\mathbf{e}_t = \mathbf{E}_{\text{embed}}[w_t] \in \mathbb{R}^D
$$

where $\mathbf{E}_{\text{embed}} \in \mathbb{R}^{V \times D}$ is a learnable embedding matrix ($V$ = vocabulary size).

This is equivalent to a one-hot lookup:

$$
\mathbf{e}_t = \mathbf{E}_{\text{embed}}^\top \mathbf{o}_t, \quad \text{where } (\mathbf{o}_t)_j = \mathbb{1}[w_t = j]
$$

#### [CLS] Token and Full Sequence

$$
\mathbf{Z}_0 = \bigl[\mathbf{e}_{\text{cls}}; \; \mathbf{e}_1; \; \mathbf{e}_2; \; \ldots; \; \mathbf{e}_T\bigr] + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(T+1) \times D}
$$

The same transformer blocks (pre-norm MSA + FFN) process $\mathbf{Z}_0$. Output:

$$
\mathbf{h}_{\text{txt}} = \text{LN}(\mathbf{Z}_L^0) \in \mathbb{R}^D
$$

The [CLS] token aggregates sequence-level semantics via self-attention — analogous to ViT's [CLS] aggregating patch-level visual information.

```
Token IDs:  [CLS]  "a"  "cat"  "sits"     →  shape flow:

(B, T+1) ──→ Embed ──→ (B, T+1, D) ──→ + pos ──→ Transformer×L ──→ [CLS] ──→ (B, D) = h_txt
```

#### Encoder Size Comparison

```
  ResNet-18     |========|                    11.7M params
  ViT-Tiny      |======|                       5.7M params
  ViT-Small     |=============|               22.0M params
  ViT-Base      |=========================|   86.0M params
  BERT-Base     |===========================| 110.0M params
  Our Mini-ViT  |=|                            0.5M params  <-- LOW COMPUTE
```

### Key Takeaway (Notebook 2)

> Both encoders map their input to a fixed-size vector $\mathbf{h} \in \mathbb{R}^D$. Shared dimensionality is the prerequisite for every fusion strategy in Notebook 3.

---

## Notebook 3: `03_fusion_strategies/03_fusion_strategies.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/03_fusion_strategies/03_fusion_strategies.ipynb)

Once we have $\mathbf{h}_{\text{img}} \in \mathbb{R}^D$ and $\mathbf{h}_{\text{txt}} \in \mathbb{R}^D$ (or full token sequences), how do we combine them? Four strategies, increasing in sophistication.

---

### 1. Early Fusion

Concatenate token sequences from both modalities **before** any cross-modal processing:

$$
\mathbf{Z}_{\text{joint}} = \bigl[\mathbf{Z}_{\text{img}}^1, \ldots, \mathbf{Z}_{\text{img}}^N, \; \mathbf{Z}_{\text{txt}}^1, \ldots, \mathbf{Z}_{\text{txt}}^T\bigr] \in \mathbb{R}^{(N+T) \times D}
$$

$$
\mathbf{Z}_{\text{out}} = \text{Transformer}(\mathbf{Z}_{\text{joint}})
$$

Every token attends to every other token — image patches see text tokens and vice versa from the first layer.

```
  Image patches:  [p1] [p2] ... [pN]  ──┐
                                         ├──→ [p1..pN, w1..wT] ──→ Transformer ──→ output
  Text tokens:    [w1] [w2] ... [wT]  ──┘
```

**Complexity:** Self-attention over $(N+T)$ tokens costs:

$$
O\bigl((N+T)^2 \cdot D\bigr)
$$

For $N = 64$, $T = 16$: $(64+16)^2 = 6400$ attention pairs per head. Expensive, but maximum interaction depth.

---

### 2. Late Fusion

Process each modality independently through its own transformer, then combine only the final representations:

$$
\mathbf{h} = g(\mathbf{h}_{\text{img}}, \mathbf{h}_{\text{txt}})
$$

Three common combination functions $g$:

| Method | Formula | Output dim | Properties |
|--------|---------|------------|------------|
| **Concatenation** | $\mathbf{h} = [\mathbf{h}_{\text{img}}; \mathbf{h}_{\text{txt}}]$ | $2D$ | Preserves all info; needs downstream layer |
| **Addition** | $\mathbf{h} = \mathbf{h}_{\text{img}} + \mathbf{h}_{\text{txt}}$ | $D$ | Requires $D$-dim alignment; parameter-free |
| **Hadamard (element-wise product)** | $\mathbf{h} = \mathbf{h}_{\text{img}} \odot \mathbf{h}_{\text{txt}}$ | $D$ | Captures multiplicative interactions |

```
  Image ──→ Encoder ──→ h_img ──┐
                                 ├──→ g(·,·) ──→ h_fused
  Text  ──→ Encoder ──→ h_txt ──┘

  No cross-modal interaction until the very end!
```

**Complexity:** $O(N^2 D + T^2 D)$ — two separate self-attention passes, no cross terms.

---

### 3. Cross-Attention — Deep Dive

Cross-attention is the most powerful fusion mechanism. One modality provides **queries**; the other provides **keys** and **values**.

#### Full Derivation (Text queries Image)

Given $\mathbf{Z}_{\text{txt}} \in \mathbb{R}^{T \times D}$ and $\mathbf{Z}_{\text{img}} \in \mathbb{R}^{N \times D}$:

$$
\mathbf{Q} = \mathbf{Z}_{\text{txt}} \mathbf{W}_Q \in \mathbb{R}^{T \times d_k}
$$

$$
\mathbf{K} = \mathbf{Z}_{\text{img}} \mathbf{W}_K \in \mathbb{R}^{N \times d_k}
$$

$$
\mathbf{V} = \mathbf{Z}_{\text{img}} \mathbf{W}_V \in \mathbb{R}^{N \times d_v}
$$

$$
\text{CrossAttn}(\mathbf{Z}_{\text{txt}}, \mathbf{Z}_{\text{img}}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V} \in \mathbb{R}^{T \times d_v}
$$

#### Attention Matrix Interpretation

The matrix $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}) \in \mathbb{R}^{T \times N}$ encodes cross-modal relevance:

$$
A_{ij} = \text{"how much text token } i \text{ attends to image patch } j\text{"}
$$

Each row sums to 1: $\sum_{j=1}^{N} A_{ij} = 1$.

#### ASCII Attention Heatmap

```
  Attention Matrix A (T×N):  "What color is the cat?"

  Text\Image    p₁    p₂    p₃    p₄   ...  p_N
  ─────────────────────────────────────────────
  "what"       0.05  0.06  0.04  0.05  ...  0.05    ← diffuse (function word)
  "color"      0.02  0.35  0.28  0.03  ...  0.03    ← attends to colorful patches
  "is"         0.04  0.05  0.05  0.04  ...  0.04
  "the"        0.03  0.04  0.05  0.03  ...  0.03
  "cat"        0.01  0.02  0.01  0.42  ...  0.08    ← attends to cat patch!

  █ = high attention (0.3+)    ░ = low attention (<0.1)

  Visual heatmap on 8×8 patch grid for token "cat":
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | ░  | ░  | ░  | ░  | ░  | ░  |
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | ░  | ░  | ░  | ░  | ░  | ░  |
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | ░  | ░  | ░  | ░  | ░  | ░  |
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | ░  | █  | █  | █  | ░  | ░  |  ← cat body patches
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | █  | █  | █  | █  | ░  | ░  |
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | ░  | █  | █  | ░  | ░  | ░  |
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | ░  | ░  | ░  | ░  | ░  | ░  |
  +----+----+----+----+----+----+----+----+
  | ░  | ░  | ░  | ░  | ░  | ░  | ░  | ░  |
  +----+----+----+----+----+----+----+----+
```

#### Bidirectional Cross-Attention

Rich multimodal models apply cross-attention in **both directions**:

$$
\mathbf{Z}_{\text{txt}}' = \text{CrossAttn}(\mathbf{Q}=\mathbf{Z}_{\text{txt}},\; \mathbf{K,V}=\mathbf{Z}_{\text{img}}) + \mathbf{Z}_{\text{txt}}
$$

$$
\mathbf{Z}_{\text{img}}' = \text{CrossAttn}(\mathbf{Q}=\mathbf{Z}_{\text{img}},\; \mathbf{K,V}=\mathbf{Z}_{\text{txt}}) + \mathbf{Z}_{\text{img}}
$$

```
  Text tokens ──Q──→ CrossAttn ←──K,V── Image patches     (text looks at image)
  Image patches ──Q──→ CrossAttn ←──K,V── Text tokens     (image looks at text)
```

**Complexity:** $O(T \cdot N \cdot D)$ per direction — linear in the product of sequence lengths, much cheaper than early fusion's $O((N+T)^2 D)$ when $T \ll N+T$.

---

### 4. Gated Fusion

A learned gate dynamically weights each modality's contribution:

$$
\boldsymbol{\alpha} = \sigma\!\left(\mathbf{W}_g [\mathbf{h}_{\text{img}}; \mathbf{h}_{\text{txt}}] + \mathbf{b}_g\right)
$$

$$
\mathbf{h}_{\text{fused}} = \boldsymbol{\alpha} \odot \mathbf{h}_{\text{img}} + (\mathbf{1} - \boldsymbol{\alpha}) \odot \mathbf{h}_{\text{txt}}
$$

#### Per-Element Gating

$\boldsymbol{\alpha} \in \mathbb{R}^D$ (not a scalar) — each embedding dimension can independently favor image or text:

```
  α = [0.9, 0.1, 0.8, 0.3, ...]   ← dimension 0 trusts image, dimension 1 trusts text
       ↓     ↓     ↓     ↓
  h_fused = α ⊙ h_img + (1-α) ⊙ h_txt   (element-wise)
```

#### Sigmoid Properties

$\sigma(z) = \frac{1}{1 + e^{-z}}$:

| Property | Value |
|----------|-------|
| Range | $(0, 1)$ |
| $\sigma(0)$ | $0.5$ (equal weighting) |
| $\sigma(z) \to 1$ as $z \to +\infty$ | Favor first modality |
| $\sigma(z) \to 0$ as $z \to -\infty$ | Favor second modality |
| Derivative | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (max at $z=0$) |

For image-heavy inputs (clear photo, vague caption): $\boldsymbol{\alpha} \to \mathbf{1}$. For text-heavy inputs: $\boldsymbol{\alpha} \to \mathbf{0}$.

---

### 5. Bilinear Fusion (Bonus)

Bilinear fusion captures **multiplicative interactions** between modalities via a 3-way weight tensor:

$$
h = \mathbf{x}_1^\top \mathbf{W} \mathbf{x}_2, \quad \mathbf{W} \in \mathbb{R}^{d_1 \times d_2 \times d_o}
$$

Equivalently, for output dimension $k$:

$$
h_k = \sum_{i=1}^{d_1} \sum_{j=1}^{d_2} W_{ijk} \, x_{1,i} \, x_{2,j}
$$

This is strictly more expressive than addition or Hadamard product but costs $O(d_1 \cdot d_2 \cdot d_o)$ parameters. Used in VQA (e.g., MCB, MLB architectures).

---

### Fusion Comparison Summary

| Strategy | Complexity | Parameters | Interaction Depth | Best For |
|----------|-----------|------------|-------------------|----------|
| **Early** | $O((N+T)^2 D)$ | Shared transformer | Maximum (full attention) | Modalities share structure; small $N, T$ |
| **Late** | $O(N^2 D + T^2 D)$ | Only $g(\cdot)$ | Minimal (final vectors only) | Quick baseline; frozen pretrained encoders |
| **Cross-Attn** | $O(TND)$ per direction | Cross-attn layers | Rich (selective, token-level) | Most tasks — **recommended** |
| **Gated** | $O(D)$ | $W_g \in \mathbb{R}^{D \times 2D}$ | Adaptive weighting | Varying modality quality/reliability |
| **Bilinear** | $O(d_1 d_2 d_o)$ | 3-way tensor | Multiplicative cross-terms | Fine-grained VQA, structured reasoning |

### Key Takeaway (Notebook 3)

> **Cross-attention** is the most powerful fusion mechanism: $\text{softmax}(QK^\top / \sqrt{d_k})V$ where $Q$ comes from one modality and $K, V$ from another. It enables selective, interpretable, token-level cross-modal reasoning.

---

## Concepts Reference

| Concept | Math | Introduced In | Used Later In |
|---------|------|---------------|---------------|
| Cosine Similarity | $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ | Notebook 01 | Module 02 (CLIP), Module 03 (InfoNCE) |
| Contrastive Loss (InfoNCE) | $-\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$ | Notebook 01 | Module 02 (CLIP training) |
| Patch Embedding | $\mathbf{x}_p^i \mathbf{E} + \mathbf{e}_{\text{pos}}^i$ | Notebook 02 | Module 02 (CLIP), Module 05 (LLaVA) |
| Sinusoidal PE | $\sin/\cos(\text{pos}/10000^{2i/d})$ | Notebook 02 | Module 04 (generative models) |
| Scaled Dot-Product Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ | Notebook 02 | All subsequent modules |
| Multi-Head Attention | $\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$ | Notebook 02 | Module 02, 03, 05 |
| Layer Normalization | $\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$ | Notebook 02 | All transformer modules |
| FFN + GELU | $W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ | Notebook 02 | All transformer modules |
| [CLS] Token | $\mathbf{h} = \text{LN}(\mathbf{Z}_L^0)$ | Notebook 02 | Module 02, 03 |
| Early Fusion | $[\mathbf{Z}_{\text{img}}; \mathbf{Z}_{\text{txt}}]$ | Notebook 03 | Module 04 |
| Late Fusion | $g(\mathbf{h}_{\text{img}}, \mathbf{h}_{\text{txt}})$ | Notebook 03 | Module 02 (baselines) |
| Cross-Attention | $\text{softmax}(Q_{\text{txt}} K_{\text{img}}^\top / \sqrt{d_k}) V_{\text{img}}$ | Notebook 03 | Module 02 (Captioning, VQA), Module 05 |
| Gated Fusion | $\alpha \odot \mathbf{h}_{\text{img}} + (1-\alpha) \odot \mathbf{h}_{\text{txt}}$ | Notebook 03 | Module 04 |
| Bilinear Fusion | $\mathbf{x}_1^\top \mathbf{W} \mathbf{x}_2$ | Notebook 03 | Module 03 (VQA) |
| Projection Head | $\mathbf{h}_{\text{proj}} = W_p \mathbf{h} + b_p$ | Notebook 02 | Module 02, 04 |

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total code cells | ~20 |
| Total visualizations | 12 |
| Max model size | ~500K params |
| Runs on CPU | Yes (all of it) |
| Data downloads needed | None (synthetic data) |
| Key equations covered | 40+ |
| Tensor shape annotations | Full ViT + Text pipeline |

---

## Next Step

Now that you understand encoders and fusion, it's time to build real models:

**[02_Vision_Language_Models/01_clip_from_scratch/01_clip_from_scratch.ipynb](../02_Vision_Language_Models/01_clip_from_scratch/01_clip_from_scratch.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/01_clip_from_scratch/01_clip_from_scratch.ipynb)

---

## Notebooks 4–6 (New — Prerequisite Deep Dives)

| # | Notebook | Topic |
|---|----------|-------|
| 04 | [04_attention_mechanism](04_attention_mechanism/README.md) | Self-attention, multi-head, cross-attention, complexity |
| 05 | [05_transformer_architecture](05_transformer_architecture/README.md) | Full encoder-decoder, positional encoding, causal mask |
| 06 | [06_tokenization_embeddings](06_tokenization_embeddings/README.md) | BPE, WordPiece, patch embedding |
