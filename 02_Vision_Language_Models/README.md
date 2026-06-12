# Module 02: Vision-Language Models

> **Time:** 2–3 hours | **Notebooks:** 3 | **Visuals:** 8 plots | **Difficulty:** Intermediate

| Notebook | Open in Colab |
|----------|---------------|
| `01_clip_from_scratch.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/01_clip_from_scratch.ipynb) |
| `02_image_captioning.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/02_image_captioning.ipynb) |
| `03_visual_question_answering.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/03_visual_question_answering.ipynb) |

---

## The Story So Far

In Module 01, you learned how encoders convert images and text into vectors, and how fusion strategies combine them. Now we build **real models** that solve real tasks — with full mathematical machinery.

This module covers the **three pillars** of vision-language AI:

```
+------------------------------------------------------------------+
|                                                                  |
|           The 3 Vision-Language Paradigms                        |
|                                                                  |
|  +-----------------+  +-----------------+  +-----------------+   |
|  |   CONTRASTIVE   |  |   GENERATIVE    |  | DISCRIMINATIVE  |   |
|  |                 |  |                 |  |                 |   |
|  |    CLIP         |  |   Captioning    |  |     VQA         |   |
|  |                 |  |                 |  |                 |   |
|  |  "Does this     |  |  "Describe      |  |  "What color    |   |
|  |   image match   |  |   this image    |  |   is the cat?"  |   |
|  |   this text?"   |  |   in words"     |  |   -> "orange"   |   |
|  +-----------------+  +-----------------+  +-----------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

| Model | Type | Params | What It Does |
|-------|------|--------|-------------|
| **Mini-CLIP** | Contrastive | ~1.5M | Matches images to text descriptions |
| **Image Captioner** | Generative | ~1.2M | Generates text descriptions from images |
| **VQA Model** | Discriminative | ~1.0M | Answers questions about images |

---

## Notebook 1: `01_clip_from_scratch.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/01_clip_from_scratch.ipynb)

### 1. The Full CLIP Pipeline

Given a batch of $N$ image–text pairs $\{(I_i, T_i)\}_{i=1}^N$, CLIP learns two encoders that map images and text into a **shared embedding space** where matched pairs are close and unmatched pairs are far apart.

```
  IMAGE PATH                                    TEXT PATH
  ==========                                    ==========

  +------------------+                    +------------------+
  |  Input Image I   |                    |  Input Text T    |
  |  (H x W x 3)     |                    |  token seq L     |
  +--------+---------+                    +--------+---------+
           |                                       |
           v                                       v
  +------------------+                    +------------------+
  |  ViT / CNN       |                    |  Transformer     |
  |  Encoder f_I     |                    |  Encoder f_T     |
  |  -> v in R^D     |                    |  -> t in R^D     |
  +--------+---------+                    +--------+---------+
           |                                       |
           v                                       v
  +------------------+                    +------------------+
  |  Projection Head |                    |  Projection Head |
  |  W_proj^I        |                    |  W_proj^T        |
  |  R^D -> R^d      |                    |  R^D -> R^d      |
  +--------+---------+                    +--------+---------+
           |                                       |
           v                                       v
  +------------------+                    +------------------+
  |  L2 Normalize    |                    |  L2 Normalize    |
  |  v -> v_hat      |                    |  t -> t_hat      |
  +--------+---------+                    +--------+---------+
           |                                       |
           +--------->  S = V_hat @ T_hat^T  <----+
                              |
                              v
                    +------------------+
                    |  InfoNCE Loss    |
                    |  (symmetric)     |
                    +------------------+
                              |
                              v
                    +------------------+
                    |  Zero-Shot       |
                    |  Classification  |
                    +------------------+
```

Stack the batch into matrices $\mathbf{V}, \mathbf{T} \in \mathbb{R}^{N \times d}$ (rows are embeddings). After normalization:

$$\mathbf{S} = \hat{\mathbf{V}} \hat{\mathbf{T}}^\top \in \mathbb{R}^{N \times N}, \qquad S_{ij} = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j$$

---

### 2. The Projection Head — Why Learn a Projection?

Raw encoder outputs $\mathbf{v} = f_I(I) \in \mathbb{R}^D$ live in modality-specific spaces. A **learned linear projection** maps both modalities into a shared contrastive space of dimension $d$:

$$\mathbf{v}' = W_{\text{proj}}^I \, \mathbf{v}, \qquad \mathbf{t}' = W_{\text{proj}}^T \, \mathbf{t}$$

where $W_{\text{proj}}^I, W_{\text{proj}}^T \in \mathbb{R}^{d \times D}$.

**Why not use encoder outputs directly?**

1. **Dimensionality alignment** — Image and text encoders may have different hidden sizes ($D_I \neq D_T$); projection unifies them to $d$.
2. **Task-specific rotation** — Encoders were pretrained for reconstruction/classification; projection learns the rotation that makes cosine similarity meaningful for matching.
3. **Decoupling capacity** — Encoders capture rich features; the projection head specializes in the matching objective without destroying pretrained representations.

After projection, L2 normalization is applied:

$$\hat{\mathbf{v}} = \frac{W_{\text{proj}}^I \mathbf{v}}{\|W_{\text{proj}}^I \mathbf{v}\|_2}, \qquad \hat{\mathbf{t}} = \frac{W_{\text{proj}}^T \mathbf{t}}{\|W_{\text{proj}}^T \mathbf{t}\|_2}$$

The notation $\mathbf{v}' = W_{\text{proj}} \mathbf{v} / \|W_{\text{proj}} \mathbf{v}\|$ combines projection and normalization in one step.

---

### 3. L2 Normalization — Geometry on the Unit Hypersphere

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \qquad \|\hat{\mathbf{v}}\|_2 = 1$$

**Geometric meaning:** Every embedding lies on the $(d-1)$-dimensional unit hypersphere $\mathbb{S}^{d-1} = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\|_2 = 1\}$.

Consequences:

- **Similarity reduces to angle.** For unit vectors, $\hat{\mathbf{v}}^\top \hat{\mathbf{t}} = \cos\theta$ where $\theta$ is the angle between them.
- **Range is bounded:** $S_{ij} \in [-1, 1]$.
- **Scale invariance:** Multiplying $\mathbf{v}$ by any $\alpha > 0$ does not change $\hat{\mathbf{v}}$. The model cannot cheat by making all embeddings large.
- **Hypersphere clustering:** InfoNCE pulls matched pairs toward the same region of $\mathbb{S}^{d-1}$ and pushes negatives apart along geodesics (great-circle distances).

```
                    *  t_2 (unmatched)
                   /
                  /  theta_12
                 /
    v_1 --------+--------> t_1 (matched, small angle)
               /
              *
             t_3

    All vectors lie on the surface of a sphere.
    Matched pairs: small angle (cos ~ 1)
    Unmatched pairs: large angle (cos ~ 0 or negative)
```

---

### 4. Cosine Similarity Matrix

Stack normalized embeddings:

$$\hat{\mathbf{V}} = \begin{bmatrix} \hat{\mathbf{v}}_1^\top \\ \vdots \\ \hat{\mathbf{v}}_N^\top \end{bmatrix} \in \mathbb{R}^{N \times d}, \qquad \hat{\mathbf{T}} = \begin{bmatrix} \hat{\mathbf{t}}_1^\top \\ \vdots \\ \hat{\mathbf{t}}_N^\top \end{bmatrix} \in \mathbb{R}^{N \times d}$$

The similarity matrix is a single matrix multiply:

$$\mathbf{S} = \hat{\mathbf{V}} \hat{\mathbf{T}}^\top \in \mathbb{R}^{N \times N}, \qquad S_{ij} = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j = \cos(\angle(\hat{\mathbf{v}}_i, \hat{\mathbf{t}}_j))$$

| Entry | Meaning |
|-------|---------|
| $S_{ii}$ (diagonal) | Similarity of matched pair $(I_i, T_i)$ — should be **maximized** |
| $S_{ij}$, $i \neq j$ (off-diagonal) | Similarity of unmatched pair — should be **minimized** |

For $N=4$, $\mathbf{S}$ is $4 \times 4$ with positive diagonal dominance when training is working.

---

### 5. InfoNCE Loss — Full Worked Numerical Example ($N=4$, $\tau=0.07$)

Consider a mini-batch of $N=4$ image–text pairs. After encoding and normalization, suppose the raw similarity matrix is:

$$
\mathbf{S} = \begin{bmatrix}
0.50 & 0.10 & 0.08 & 0.06 \\
0.12 & 0.55 & 0.09 & 0.07 \\
0.07 & 0.11 & 0.52 & 0.08 \\
0.05 & 0.08 & 0.10 & 0.48
\end{bmatrix}
$$

Each row $i$ lists similarities between image $I_i$ and all text captions $T_1, \ldots, T_4$. The diagonal $(0.50, 0.55, 0.52, 0.48)$ are the **correct** pairs.

#### Step A: Scale by temperature

$$\frac{\mathbf{S}}{\tau} = \frac{\mathbf{S}}{0.07} = \begin{bmatrix}
7.143 & 1.429 & 1.143 & 0.857 \\
1.714 & 7.857 & 1.286 & 1.000 \\
1.000 & 1.571 & 7.429 & 1.143 \\
0.714 & 1.143 & 1.429 & 6.857
\end{bmatrix}$$

Low $\tau$ amplifies differences — the correct match (7.143) dominates the row.

#### Step B: Row-wise softmax — full computation for row 1

Row 1 scaled logits: $[7.143,\; 1.429,\; 1.143,\; 0.857]$.

Compute exponentials:

| $j$ | $S_{1j}/\tau$ | $\exp(S_{1j}/\tau)$ |
|-----|---------------|---------------------|
| 1 | 7.143 | 1265.04 |
| 2 | 1.429 | 4.17 |
| 3 | 1.143 | 3.14 |
| 4 | 0.857 | 2.36 |

Row sum: $Z_1 = 1265.04 + 4.17 + 3.14 + 2.36 = 1274.71$

Softmax probabilities for row 1:

$$p(T_j \mid I_1) = \frac{\exp(S_{1j}/\tau)}{\sum_{k=1}^{4} \exp(S_{1k}/\tau)}$$

| $j$ | $p(T_j \mid I_1)$ |
|-----|-------------------|
| 1 | $1265.04 / 1274.71 = 0.9924$ |
| 2 | $4.17 / 1274.71 = 0.0033$ |
| 3 | $3.14 / 1274.71 = 0.0025$ |
| 4 | $2.36 / 1274.71 = 0.0019$ |

Loss contribution from pair 1:

$$-\log p(T_1 \mid I_1) = -\log(0.9924) = 0.0076$$

#### Step C: All diagonal $-\log$ terms (image → text)

Applying the same row-wise softmax to all rows:

| Pair $i$ | $S_{ii}$ | $p(T_i \mid I_i)$ | $-\log p(T_i \mid I_i)$ |
|----------|----------|-------------------|-------------------------|
| 1 | 0.50 | 0.9924 | **0.0076** |
| 2 | 0.55 | 0.9954 | **0.0046** |
| 3 | 0.52 | 0.9937 | **0.0063** |
| 4 | 0.48 | 0.9903 | **0.0098** |

$$\mathcal{L}_{I \to T} = \frac{1}{4}(0.0076 + 0.0046 + 0.0063 + 0.0098) = \frac{0.0283}{4} = \mathbf{0.0071}$$

#### Step D: Text → image direction (column-wise softmax)

For $\mathcal{L}_{T \to I}$, softmax is applied **across rows** for each column $j$ (given text $T_j$, which image matches?):

$$\mathcal{L}_{T \to I} = -\frac{1}{N}\sum_{j=1}^{N} \log \frac{\exp(S_{jj}/\tau)}{\sum_{i=1}^{N} \exp(S_{ij}/\tau)} = \mathbf{0.0070}$$

#### Step E: Final symmetric loss

$$\mathcal{L}_{\text{CLIP}} = \frac{\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}}{2} = \frac{0.0071 + 0.0070}{2} = \boxed{0.0070}$$

This is a **well-trained batch** — loss near zero means matched pairs dominate their rows/columns. An untrained model with uniform similarities ($S_{ij} \approx 0.25$) would give $\mathcal{L} \approx -\log(0.25) = 1.39$.

---

### 6. Symmetric Loss — Why Both Directions Matter

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\right)$$

where:

$$\mathcal{L}_{I \to T} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N} \exp(S_{ij}/\tau)}$$

$$\mathcal{L}_{T \to I} = -\frac{1}{N}\sum_{j=1}^{N} \log \frac{\exp(S_{jj}/\tau)}{\sum_{i=1}^{N} \exp(S_{ij}/\tau)}$$

**Why both?**

| Direction | Question asked | Without it |
|-----------|----------------|------------|
| $I \to T$ | Given image, find correct text | Image encoder may collapse — many images map to same point |
| $T \to I$ | Given text, find correct image | Text encoder may collapse — many captions map to same point |

Asymmetric training lets one encoder free-ride: e.g., $\mathcal{L}_{I \to T}$ alone could be minimized if all text embeddings cluster together (making softmax easy) while image embeddings spread arbitrarily. Symmetry enforces **mutual alignment** — both encoders must organize $\mathbb{S}^{d-1}$ consistently.

Equivalently, this is two cross-entropy losses with label $y_i = i$:

$$\mathcal{L}_{I \to T} = \text{CrossEntropy}(\mathbf{S}/\tau,\; [0,1,\ldots,N-1])$$

---

### 7. Temperature $\tau$ — Analysis, Learnable Parameter, and Gradient

#### Limits

Write scaled logits as $z_{ij} = S_{ij}/\tau$.

**As $\tau \to 0$:** $z_{ij} \to \infty$ for the largest $S_{ij}$ in each row, so softmax → **one-hot** (hard argmax). Gradients vanish for non-winners — optimization becomes brittle but decisions are sharp.

**As $\tau \to \infty$:** $z_{ij} \to 0$, so softmax → **uniform** $1/N$. All pairs look equally likely; loss $\to \log N$. Gradients are equal across all negatives — no contrastive signal.

```
  tau -> 0          tau = 0.07 (CLIP)       tau -> inf
  =========         =================       ==========
  sharp peaks       useful gradients        flat uniform
  hard decisions    good separation           no learning signal
  loss -> 0         loss ~ 0.01-1.0           loss -> log(N)
```

#### Learnable temperature in CLIP

CLIP parameterizes temperature in log-space for stability:

$$\tau = \exp(t), \qquad t \in \mathbb{R} \text{ learned during training}$$

Initialized at $t = \log(1/0.07) \approx 2.66$, so $\tau_0 = 0.07$. This ensures $\tau > 0$ always without constrained optimization.

#### Gradient w.r.t. temperature

For a single pair with logits $z_i = S_{ii}/\tau$ and $z_j = S_{ij}/\tau$, the softmax probability is $p_i = e^{z_i} / Z$. The loss $\ell = -\log p_i$ has:

$$\frac{\partial \ell}{\partial \tau} = \frac{1}{\tau^2}\sum_{j=1}^{N} S_{ij}\,(p_j - \mathbb{1}[j=i])$$

**Intuition:** If high-similarity negatives ($S_{ij}$ large for $j \neq i$) receive large probability mass, the gradient pushes $\tau$ **down** (sharper distribution, penalize confusion harder). If the model is already confident, gradient near zero.

---

### 8. Zero-Shot Classification

At inference, CLIP requires **no fine-tuning** on the target dataset. Given image $I$ and $K$ class descriptions $\{T_1, \ldots, T_K\}$ (e.g., *"a photo of a dog"*, *"a photo of a cat"*):

$$\hat{y} = \arg\max_{k \in \{1,\ldots,K\}} \text{sim}\bigl(f_I(I),\; f_T(T_k)\bigr) = \arg\max_k \; \hat{\mathbf{v}}^\top \hat{\mathbf{t}}_k$$

**Pipeline:**

```
  Image I  --->  f_I  --->  v_hat  ---+
                                       +-->  cos sim with each t_hat_k  --->  argmax  --->  class
  Text T_k --->  f_T  --->  t_hat_k --+     (K similarities)
```

**Why it works:** Training on millions of $(I, T)$ pairs teaches $f_I$ and $f_T$ to align visual concepts with linguistic descriptions. At test time, class names are just new text prompts — no gradient updates needed.

---

### 9. SigLIP Comparison (Bonus)

SigLIP replaces row-wise softmax (which couples all samples in a batch) with **pairwise sigmoid** losses:

$$\mathcal{L}_{\text{SigLIP}} = -\sum_{i=1}^{N}\sum_{j=1}^{N} \log \sigma(z_{ij} \cdot m_{ij})$$

where $z_{ij} = S_{ij}/\tau$, $\sigma$ is the sigmoid function, and the label matrix is:

$$m_{ij} = 2\,\mathbb{1}[i=j] - 1 = \begin{cases} +1 & \text{if } i = j \text{ (matched)} \\ -1 & \text{if } i \neq j \text{ (unmatched)} \end{cases}$$

| Property | InfoNCE (CLIP) | SigLIP |
|----------|----------------|--------|
| Normalization | Row softmax (partition function) | Independent sigmoids |
| Batch coupling | Strong — all $N$ pairs compete | Weak — each pair treated separately |
| Large batch need | Yes (more negatives) | Less critical |
| Loss form | $\mathcal{L} = -\log \frac{e^{z_{ii}}}{\sum_j e^{z_{ij}}}$ | $\mathcal{L} = -\sum_{i,j} \log \sigma(z_{ij} m_{ij})$ |

---

### Key Numbers (Notebook 1)

| Metric | Value |
|--------|-------|
| Total parameters | ~1.5M |
| Projection dim $d$ | 64 |
| Temperature $\tau$ | 0.07 (learnable, $\tau = e^t$) |
| Training data | 200 synthetic pairs |
| Training time (CPU) | ~10 seconds |

---

## Notebook 2: `02_image_captioning.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/02_image_captioning.ipynb)

### 1. Conditional Language Modeling

Image captioning is **conditional language modeling** — generate caption $\mathbf{y} = (y_1, y_2, \ldots, y_T)$ token by token:

$$P(\mathbf{y} \mid I) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, I)$$

By the chain rule of probability, the joint distribution factorizes into next-token predictions, each conditioned on the image $I$ and all prior tokens $y_{<t} = (y_1, \ldots, y_{t-1})$.

At step $t$, the model outputs a distribution over vocabulary $\mathcal{V}$:

$$P(y_t = w \mid y_{<t}, I) = \text{softmax}(\mathbf{W}_{\text{out}} \mathbf{h}_t + \mathbf{b})_w, \qquad w \in \mathcal{V}$$

where $\mathbf{h}_t$ is the decoder hidden state at position $t$.

---

### 2. Encoder–Decoder Architecture

```
  ENCODER (Vision)                         DECODER (Language)
  =================                         ==================

  +------------------+                     +------------------+
  |  Image I         |                     |  Token seq       |
  |  H x W x 3       |                     |  [BOS] y1 ... yT |
  +--------+---------+                     +--------+---------+
           |                                        |
           v                                        v
  +------------------+                     +------------------+
  |  CNN / ViT       |                     |  Token Embedding |
  |  patch features  |                     |  + Pos Encoding  |
  +--------+---------+                     +--------+---------+
           |                                        |
           v                                        v
  +------------------+                     +------------------+
  |  Z_img           |                     |  Causal Self-Attn|
  |  (N_p x d)       |                     |  (masked)        |
  |  N_p patches     |                     +--------+---------+
  +--------+---------+                              |
           |                                        v
           |                               +------------------+
           +------------------------------>|  Cross-Attention |
                                           |  Q=Z_txt, KV=Z_img|
                                           +--------+---------+
                                                    |
                                                    v
                                           +------------------+
                                           |  Feed-Forward    |
                                           +--------+---------+
                                                    |
                                                    v
                                           +------------------+
                                           |  Linear + Softmax|
                                           |  -> P(y_t|...)  |
                                           +------------------+

  Dimensions:
    Z_img  in R^{N_p x d}     (N_p image patch tokens)
    Z_txt  in R^{T   x d}     (T text tokens)
    Q      in R^{T   x d_k}   (from text)
    K,V    in R^{N_p x d_k}   (from image)
```

---

### 3. Causal Masking

Autoregressive generation requires that position $t$ cannot attend to positions $> t$. The causal mask $\mathbf{M} \in \mathbb{R}^{T \times T}$:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \quad \text{(can attend)} \\ -\infty & \text{if } i < j \quad \text{(blocked)} \end{cases}$$

Attention becomes:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

**Example ($T=5$):** rows = query position, cols = key position.

```
         key:   1      2      3      4      5
              +------+------+------+------+------+
  query 1     |  0   | -inf | -inf | -inf | -inf |
  query 2     |  0   |  0   | -inf | -inf | -inf |
  query 3     |  0   |  0   |  0   | -inf | -inf |
  query 4     |  0   |  0   |  0   |  0   | -inf |
  query 5     |  0   |  0   |  0   |  0   |  0   |
              +------+------+------+------+------+

  Position 3 can see tokens 1,2,3 only (lower triangle).
```

After softmax, blocked positions receive weight 0 — no information leakage from the future.

---

### 4. Cross-Attention in the Decoder — Dimensions

Cross-attention fuses image features into text generation. **Queries from text; keys and values from image:**

$$\mathbf{Q} = \mathbf{Z}_{\text{txt}} \mathbf{W}_Q \in \mathbb{R}^{T \times d_k}, \quad \mathbf{K} = \mathbf{Z}_{\text{img}} \mathbf{W}_K \in \mathbb{R}^{N_p \times d_k}, \quad \mathbf{V} = \mathbf{Z}_{\text{img}} \mathbf{W}_V \in \mathbb{R}^{N_p \times d_v}$$

$$\text{CrossAttn}(\mathbf{Z}_{\text{txt}}, \mathbf{Z}_{\text{img}}) = \underbrace{\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)}_{\mathbf{A} \in \mathbb{R}^{T \times N_p}} \mathbf{V} \in \mathbb{R}^{T \times d_v}$$

| Tensor | Shape | Role |
|--------|-------|------|
| $\mathbf{Z}_{\text{txt}}$ | $T \times d$ | Decoder token representations |
| $\mathbf{Z}_{\text{img}}$ | $N_p \times d$ | Encoder patch representations |
| $\mathbf{A}$ | $T \times N_p$ | Attention weights — which patches each word looks at |
| Output | $T \times d_v$ | Image-informed text representations |

Entry $A_{t,p}$ = how much token $t$ attends to image patch $p$. When generating *"cat"*, $A_{t,\cdot}$ should peak on patches containing the cat.

---

### 5. Teacher Forcing

During **training**, the decoder input is the **ground-truth** caption shifted right by one:

$$\text{input}_t = \begin{cases} \text{[BOS]} & t = 1 \\ y_{t-1} & t > 1 \end{cases}$$

Loss (cross-entropy at each position):

$$\mathcal{L}_{\text{caption}} = -\sum_{t=1}^{T} \log P(y_t \mid y_1, \ldots, y_{t-1}, I) = \frac{1}{T}\sum_{t=1}^{T} \text{CE}(\text{logits}_t, y_t)$$

**Why teacher forcing?** Parallel computation over all $T$ positions — no sequential decoding loop during training. The model always sees correct prefixes, so gradients are stable.

**Exposure bias:** At inference, the model sees its own (possibly wrong) predictions. Mitigations: scheduled sampling, beam search.

---

### 6. Decoding Strategies

#### Greedy decoding

$$y_t = \arg\max_{w \in \mathcal{V}} P(w \mid y_{<t}, I)$$

Fast ($O(T)$ forward passes) but suboptimal — one wrong token derails the entire caption.

#### Beam search

Maintain top-$B$ partial hypotheses. Score:

$$\text{score}(\mathbf{y}) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, I)$$

```
  t=1:  [("a", -0.3), ("the", -0.5), ("one", -1.2)]   <- top B=3
  t=2:  [("a cute", -0.8), ("a small", -1.1), ("the cat", -1.0)]
  t=3:  [("a cute cat", -1.2), ...]
  ...
  Return highest-scoring complete sequence.
```

Complexity $O(B \cdot T \cdot |\mathcal{V}|)$ — much better quality, slower.

#### Top-$k$ sampling

At each step, restrict to the $k$ highest-probability tokens, renormalize, sample:

$$P'(w) = \frac{P(w \mid y_{<t}, I)}{\sum_{w' \in \text{Top-}k} P(w' \mid y_{<t}, I)}, \quad w \in \text{Top-}k$$

Adds diversity; $k=1$ reduces to greedy.

#### Nucleus (top-$p$) sampling

Sample from the **smallest** set of tokens whose cumulative probability $\geq p$:

$$V_p = \min \left\{ V' \subseteq \mathcal{V} : \sum_{w \in V'} P(w \mid y_{<t}, I) \geq p \right\}$$

Adapts set size to model confidence — narrow when peaked, wide when uncertain.

---

### 7. Evaluation Metrics

#### BLEU (Bilingual Evaluation Understudy)

$$\text{BLEU-N} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ = modified $n$-gram precision, $w_n = 1/N$ (uniform), and brevity penalty:

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$$

$c$ = candidate length, $r$ = reference length. Penalizes overly short captions.

#### CIDEr (Consensus-Based Image Description Evaluation)

TF-IDF weighted $n$-gram similarity between candidate and reference captions:

$$\text{CIDEr}_n(c, S) = \frac{1}{m}\sum_j \frac{g^n(c) \cdot g^n(s_j)}{\|g^n(c)\|\,\|g^n(s_j)\|}$$

where $g^n(\cdot)$ is the TF-IDF vector of $n$-grams, $S = \{s_j\}$ are reference captions. Higher weight on rare, descriptive $n$-grams that match human consensus.

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

Alignment-based: finds best word-to-word mapping between candidate and reference using stemming and synonymy (WordNet). Combines precision, recall, and fragmentation penalty:

$$F_{\text{mean}} = \frac{10 \cdot P \cdot R}{R + 9P}$$

$$\text{METEOR} = F_{\text{mean}} \cdot (1 - \text{Penalty})$$

More correlated with human judgment than BLEU alone for captioning.

---

## Notebook 3: `03_visual_question_answering.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/03_visual_question_answering.ipynb)

### 1. VQA as Classification

Given image $I$, question $Q$, predict answer $a$ from a fixed set $\{a_1, \ldots, a_K\}$:

$$P(a \mid I, Q) = \text{softmax}\bigl(\mathbf{W}_a \, \mathbf{h}_{\text{fused}} + \mathbf{b}\bigr)_a, \qquad \mathbf{W}_a \in \mathbb{R}^{K \times d}$$

$$\hat{a} = \arg\max_a P(a \mid I, Q)$$

Training loss:

$$\mathcal{L}_{\text{VQA}} = -\log P(a^* \mid I, Q)$$

where $a^*$ is the ground-truth answer. This is standard multi-class cross-entropy over $K$ answer classes.

---

### 2. Cross-Attention Fusion — Full Math with Dimensions

Encode separately, then fuse via cross-attention (question attends to image):

$$\mathbf{Z}_I = \text{Enc}_I(I) \in \mathbb{R}^{N_p \times d}, \qquad \mathbf{Z}_Q = \text{Enc}_Q(Q) \in \mathbb{R}^{T_q \times d}$$

$$\mathbf{Q} = \mathbf{Z}_Q \mathbf{W}_Q \in \mathbb{R}^{T_q \times d_k}, \quad \mathbf{K} = \mathbf{Z}_I \mathbf{W}_K \in \mathbb{R}^{N_p \times d_k}, \quad \mathbf{V} = \mathbf{Z}_I \mathbf{W}_V \in \mathbb{R}^{N_p \times d_v}$$

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{T_q \times N_p}, \qquad \mathbf{H}_{\text{fused}} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{T_q \times d_v}$$

Pool $\mathbf{H}_{\text{fused}}$ to a single vector $\mathbf{h}_{\text{fused}} \in \mathbb{R}^d$, then classify:

$$P(a \mid I, Q) = \text{softmax}(\mathbf{W}_a \, \mathbf{h}_{\text{fused}} + \mathbf{b})$$

**Dimension walkthrough** (typical values):

```
  Z_I:  (N_p=49, d=512)   -- 7x7 ViT patches
  Z_Q:  (T_q=20, d=512)   -- question tokens
  Q:    (20, 64)           -- d_k=64
  K:    (49, 64)
  V:    (49, 64)
  A:    (20, 49)           -- question word x image patch
  H:    (20, 64)
  pool -> h_fused: (512,)
  W_a:  (K=3000, 512)     -- 3000 candidate answers
```

---

### 3. Attention Visualization — What the Weights Mean

$\mathbf{A} \in \mathbb{R}^{T_q \times N_p}$: row $t$ shows which image patches question token $t$ focuses on.

```
Question: "What color is the cat?"

  Patch grid (7x7 flattened to N_p=49; shown as 5 cols for clarity):

  Token     Patch Attention (darker = higher weight)
  ------    ----------------------------------------
  "what"    [ .  .  .  .  . ]   uniform — function word
  "color"   [ .  ## ## .  . ]   attends to colored regions
  "is"      [ .  .  .  .  . ]   uniform
  "the"     [ .  .  .  .  . ]   uniform
  "cat"     [ ## ### ## .  . ]   attends to cat patches
  "?"       [ .  .  .  .  . ]   uniform

  ASCII heatmap for "cat" token (7x7):
       col:  1   2   3   4   5   6   7
  row 1 [ .   .   .   .   .   .   .  ]
  row 2 [ .   .   #   #   #   .   .  ]
  row 3 [ .   #   #   #   #   #   .  ]
  row 4 [ .   .   #   #   #   .   .  ]
  row 5 [ .   .   .   #   .   .   .  ]
  row 6 [ .   .   .   .   .   .   .  ]
  row 7 [ .   .   .   .   .   .   .  ]

  # = high attention on cat body
```

Interpretation: content words (*color*, *cat*) learn to ground in visual regions; function words stay diffuse.

---

### 4. Multimodal Pooling

After cross-attention, $\mathbf{H}_{\text{fused}} \in \mathbb{R}^{T_q \times d}$ must become one vector for classification.

#### Mean pooling

$$\mathbf{h} = \frac{1}{T_q}\sum_{t=1}^{T_q} \mathbf{h}_t$$

Simple, treats all question tokens equally. Can dilute signal from key tokens.

#### [CLS] token pooling

Add a learnable [CLS] token at position 0; use its output:

$$\mathbf{h} = \mathbf{h}_0$$

The [CLS] token learns to aggregate via self-attention over all question–image interactions.

#### Attention-weighted pooling

Learn scalar weights per token:

$$\alpha_t = \frac{\exp(\mathbf{w}^\top \mathbf{h}_t)}{\sum_{t'=1}^{T_q} \exp(\mathbf{w}^\top \mathbf{h}_{t'})}, \qquad \mathbf{h} = \sum_{t=1}^{T_q} \alpha_t \, \mathbf{h}_t$$

Soft selection — important tokens (e.g., *cat*, *color*) receive higher $\alpha_t$.

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| Mean | $\frac{1}{T}\sum \mathbf{h}_t$ | Simple, no extra params | Uniform weighting |
| [CLS] | $\mathbf{h}_0$ | Learned aggregation | Extra token |
| Attn pool | $\sum \alpha_t \mathbf{h}_t$ | Soft focus | Extra params ($\mathbf{w}$) |

---

### 5. VQA Approaches — Comparison

```
  +-------------------+-------------------+-------------------+
  |  CLASSIFICATION   |   GENERATIVE      |   LLM-BASED       |
  +-------------------+-------------------+-------------------+
  | Fixed answer set  | Open vocabulary   | Open vocabulary   |
  | K = 3000 classes  | seq2seq decode    | prompt + generate |
  |                   |                   |                   |
  | P(a|I,Q) =        | P(a|I,Q) =        | "Question: ...    |
  |  softmax(W h)     |  prod P(w_t|...)  |  Answer:" -> LLM  |
  |                   |                   |                   |
  | Fast inference    | Flexible answers  | Best language     |
  | Limited answers   | Slower            | understanding     |
  | VQA v1/v2 classic | GPT-4V style      | Flamingo, BLIP-2  |
  +-------------------+-------------------+-------------------+
```

| Approach | Output space | Loss | Example models |
|----------|-------------|------|----------------|
| **Classification** | $a \in \{1,\ldots,K\}$ | $-\log P(a^* \mid I, Q)$ | SAN, BAN, this notebook |
| **Generative** | Free-text sequence | $-\sum_t \log P(w_t \mid \ldots)$ | mBART-VQA, VL-T5 |
| **LLM-based** | Prompt-conditioned generation | LM loss / instruction tuning | GPT-4V, LLaVA, Flamingo |

This notebook implements **classification VQA** — appropriate when answers come from a closed set (e.g., VQA v2: *yes/no*, numbers, common objects).

---

## Architecture Comparison

| Model | Loss Function | Input $\to$ Output |
|-------|---------------|-------------------|
| **CLIP** | $\mathcal{L} = \frac{1}{2N}\displaystyle\sum_i \left[-\log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} - \log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}\right]$ | $(I, T) \to s \in \mathbb{R}$ |
| **Captioning** | $\mathcal{L} = -\displaystyle\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, I)$ | $I \to (y_1, \ldots, y_T)$ |
| **VQA** | $\mathcal{L} = -\log P(a^* \mid I, Q) = -\log \text{softmax}(\mathbf{W}_a \mathbf{h}_{\text{fused}})_{a^*}$ | $(I, Q) \to a \in \{1, \ldots, K\}$ |

```
                    CLIP                 Captioning              VQA
                  ========               ==========            ======

  Input:        img + txt                img only              img + question
  Encoder:      dual (ViT + Text)        CNN/ViT + Transformer dual (ViT + Text)
  Fusion:       cosine sim (S^T)         cross-attn (dec)      cross-attn (Q->I)
  Output:       scalar similarity        token sequence        class label
  Loss:         InfoNCE (symmetric)      token cross-entropy   answer cross-entropy
  Inference:    zero-shot classify       autoregressive decode argmax over K
  Key matrix:   S in R^{N x N}           A in R^{T x N_p}      A in R^{T_q x N_p}
```

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total notebooks | 3 |
| Total visualizations | 8 |
| Models built | 3 (CLIP + Captioner + VQA) |
| Max model size | ~1.5M params |
| Runs on CPU | Yes |
| Worked examples | InfoNCE ($N=4$, $\mathcal{L}=0.0070$), causal mask ($5\times5$), cross-attn dimensions |

---

## Next Step

You've built the models. Now learn how to **train them properly**:

**[03_Training_Strategies/01_contrastive_learning.ipynb](../03_Training_Strategies/01_contrastive_learning.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning.ipynb) — this is where the real "how to train" learning begins.
