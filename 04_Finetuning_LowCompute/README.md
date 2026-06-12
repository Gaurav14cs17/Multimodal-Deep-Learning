# Module 04: Finetuning for Low Compute

> **Time:** 3-4 hours | **Notebooks:** 4 | **Visuals:** 14 plots | **Difficulty:** Advanced
>
> **This is a CORE FOCUS module** — the most practical skill in modern AI: making large models work on small hardware.

| Notebook | Open in Colab |
|----------|---------------|
| `01_lora_from_scratch.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/01_lora_from_scratch.ipynb) |
| `02_qlora_4bit_finetuning.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/02_qlora_4bit_finetuning.ipynb) |
| `03_adapter_methods.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/03_adapter_methods.ipynb) |
| `04_finetune_clip_custom_data.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/04_finetune_clip_custom_data.ipynb) |

---

## The Problem

Full finetuning of a 7B model requires storing three copies of the parameter tensor in GPU memory during training: the weights themselves, their gradients, and the Adam optimizer's first- and second-moment estimates. Each fp32 parameter occupies 4 bytes, and Adam maintains two additional fp32 buffers per parameter.

$$\text{Memory} = \underbrace{4 \times 7\text{B}}_{\text{weights (fp32)}} + \underbrace{2 \times 4 \times 7\text{B}}_{\text{Adam states } (m, v)} + \underbrace{4 \times 7\text{B}}_{\text{gradients}} = 112 \text{ GB}$$

Your GPU has 8–15 GB. **You literally cannot do it.**

```
  +--------------------------------------------------+
  |   FULL FINETUNING of a 7B model:                 |
  |                                                    |
  |   Model weights (fp32):    28 GB                  |
  |   Optimizer states (m,v):  56 GB                  |
  |   Gradients:                28 GB                  |
  |   ─────────────────────────────────                |
  |   TOTAL:                   112 GB GPU VRAM         |
  |                                                    |
  |   Your GPU:                 8 GB (laptop)          |
  |   Gap:                    ~100 GB short            |
  +--------------------------------------------------+
```

The rest of this module develops the mathematics of **Parameter-Efficient Fine-Tuning (PEFT)**: constrain or compress the adaptation so that a 7B model trains on a single consumer GPU.

---

## Notebook 1: `01_lora_from_scratch.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/01_lora_from_scratch.ipynb)

### 1. Intrinsic Dimensionality — Why Low-Rank Updates Exist

Aghajanyan et al. (2020) measured how many degrees of freedom a pretrained model actually needs to adapt to a downstream task. Define the **intrinsic dimension** $d_{\text{int}}$ as the smallest subspace dimension $d'$ such that optimizing only within a random $d'$-dimensional subspace of parameter space reaches within $\epsilon$ of full-finetuning performance:

$$\min_{\theta_0 + P\theta'} \mathcal{L}(\theta_0 + P\theta'), \quad P \in \mathbb{R}^{D \times d'}, \quad \theta' \in \mathbb{R}^{d'}$$

Their key empirical finding:

$$d_{\text{int}} \ll D_{\text{model}}$$

For RoBERTa-large ($D \approx 3.5 \times 10^8$), $d_{\text{int}} \approx 200$–$1000$ depending on the task — orders of magnitude smaller than the full parameter count. **The finetuning update $\Delta W$ lives in a low-dimensional subspace**, even though $W_0 \in \mathbb{R}^{d \times d}$ is full rank.

LoRA exploits this directly: instead of learning $\Delta W \in \mathbb{R}^{d \times d}$, learn a rank-$r$ factorization with $r \approx d_{\text{int}}/d$.

---

### 2. SVD Motivation — Any Update Is a Sum of Rank-1 Pieces

Every weight update admits a singular value decomposition:

$$\Delta W = U \Sigma V^\top = \sum_{i=1}^{\min(m,n)} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^\top$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are orthogonal, and $\Sigma = \text{diag}(\sigma_1 \geq \sigma_2 \geq \cdots \geq 0)$.

#### Worked Example: 4×4 Matrix

Consider a toy update matrix (rank 2 in practice):

$$M = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix}$$

**Step 1 — SVD.** Computing the SVD yields:

$$\sigma_1 \approx 38.49, \quad \sigma_2 \approx 2.07, \quad \sigma_3 \approx \sigma_4 \approx 10^{-15}$$

(The matrix is exactly rank 2: rows 2, 3, 4 are linear combinations of row 1.)

$$U \approx \begin{bmatrix} -0.135 & 0.822 & -0.394 & -0.394 \\ -0.317 & 0.428 & 0.562 & 0.562 \\ -0.499 & 0.034 & -0.168 & -0.168 \\ -0.681 & -0.360 & -0.360 & -0.360 \end{bmatrix}, \quad \Sigma = \text{diag}(38.49,\; 2.07,\; 0,\; 0)$$

$$V^\top \approx \begin{bmatrix} -0.230 & -0.307 & -0.385 & -0.462 \\ 0.888 & 0.231 & -0.425 & -0.231 \\ 0.408 & -0.816 & 0.408 & 0.000 \\ 0.000 & -0.408 & 0.816 & -0.408 \end{bmatrix}$$

**Step 2 — Rank-1 approximation.** Keep only the dominant singular triplet $(\sigma_1, \mathbf{u}_1, \mathbf{v}_1)$:

$$M_1 = \sigma_1 \, \mathbf{u}_1 \mathbf{v}_1^\top \approx \begin{bmatrix} 1.21 & 1.61 & 2.02 & 2.42 \\ 2.83 & 3.77 & 4.72 & 5.66 \\ 4.45 & 5.93 & 7.42 & 8.90 \\ 6.07 & 8.09 & 10.12 & 12.14 \end{bmatrix}$$

**Step 3 — Reconstruction error.**

$$\|M - M_1\|_F = \sigma_2 \approx 2.07$$

$$\frac{\|M - M_1\|_F}{\|M\|_F} = \frac{2.07}{\sqrt{38.49^2 + 2.07^2}} = \frac{2.07}{38.55} \approx 5.4\%$$

A **single rank-1 term captures 99.7% of the Frobenius energy** ($\sigma_1^2 / \sum_i \sigma_i^2 = 38.49^2 / 1482 \approx 0.997$). LoRA sets $r = 1$ (or small $r$) and learns $B \mathbf{u}_1$ and $A \mathbf{v}_1^\top$ instead of the full $M$.

```
  Full ΔW (d×d = 16 params)          Rank-1 approximation (2d = 8 params)

  [ 1   2   3   4 ]                  [ 1.2  1.6  2.0  2.4 ]
  [ 5   6   7   8 ]    ──SVD──>      [ 2.8  3.8  4.7  5.7 ]   +  error ≈ 2.07
  [ 9  10  11  12 ]                  [ 4.5  5.9  7.4  8.9 ]
  [13  14  15  16 ]                  [ 6.1  8.1 10.1 12.1 ]

  16 learned values                   8 learned values (B∈R^{4×1}, A∈R^{1×4})
```

---

### 3. Eckart–Young–Mirsky Theorem — Optimality of Truncated SVD

> **Theorem (Eckart–Young–Mirsky, 1936).** Let $M \in \mathbb{R}^{m \times n}$ have SVD $M = U\Sigma V^\top$. The best rank-$r$ approximation to $M$ in any unitarily invariant norm — in particular the Frobenius norm — is:

$$M_r = \sum_{i=1}^{r} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^\top = U_r \Sigma_r V_r^\top$$

The minimum reconstruction error is:

$$\|M - M_r\|_F = \sqrt{\sum_{i=r+1}^{\min(m,n)} \sigma_i^2}$$

**Proof sketch.** Write $M = M_r + E$ where $E = \sum_{i=r+1} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$. Then $\|E\|_F^2 = \sum_{i=r+1} \sigma_i^2$ because the singular vectors are orthonormal. Any rank-$r$ matrix $M'$ can differ from $M_r$ only by rotating energy into the discarded singular directions, which cannot reduce the Frobenius error below the sum of squared discarded singular values. $\square$

For our 4×4 example with $r=1$: $\|M - M_1\|_F = \sqrt{2.07^2 + 0^2 + 0^2} = 2.07$ — exactly $\sigma_2$.

LoRA with rank $r$ is the **learned analogue** of $M_r$: it does not compute SVD explicitly, but gradient descent finds $BA \approx \Delta W$ in the same low-rank subspace.

---

### 4. LoRA Formulation

Given frozen pretrained weights $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA parameterizes the update as:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \; A \in \mathbb{R}^{r \times d_{\text{in}}}, \; r \ll \min(d_{\text{out}}, d_{\text{in}})$$

The forward pass:

$$h = W_0 x + \frac{\alpha}{r} BAx$$

where $x \in \mathbb{R}^{d_{\text{in}}}$, $h \in \mathbb{R}^{d_{\text{out}}}$.

**Initialization** (critical for training stability):

| Matrix | Init | Purpose |
|--------|------|---------|
| $W_0$ | Pretrained (frozen) | Preserves base model behavior |
| $A$ | Kaiming uniform: $a_{ij} \sim \mathcal{U}\left(-\sqrt{6/(d_{\text{in}}+r)},\; \sqrt{6/(d_{\text{in}}+r)}\right)$ | Breaks symmetry in the low-rank branch |
| $B$ | Zeros | Ensures $\Delta W = BA = \mathbf{0}$ at step 0 |

At initialization: $h = W_0 x + \frac{\alpha}{r} \cdot \mathbf{0} \cdot x = W_0 x$ — **exact identity** with the pretrained model. No disruption at $t=0$.

The scaling $\alpha/r$ decouples the learning rate from the rank: changing $r$ does not change the effective magnitude of $\Delta W$ if $\alpha$ is held fixed (Hu et al. set $\alpha = r$ by default, giving unit scale).

---

### 5. LoRA Architecture Diagram

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         LoRA-augmented Linear Layer                      │
  │                                                                          │
  │   Input x ∈ R^{d_in}                                                    │
  │        │                                                                 │
  │        ├──────────────────────────────┐                                 │
  │        │                              │                                 │
  │        ▼                              ▼                                 │
  │   ┌─────────────┐              ┌─────────────┐                          │
  │   │     W₀      │              │      A      │   A ∈ R^{r × d_in}      │
  │   │  (FROZEN)   │              │  (TRAIN)    │   Kaiming init           │
  │   │ d_out×d_in  │              │   r × d_in  │                          │
  │   └──────┬──────┘              └──────┬──────┘                          │
  │          │                            │                                  │
  │          │  W₀x                       │  Ax  ∈ R^r  (bottleneck!)       │
  │          │                            ▼                                  │
  │          │                     ┌─────────────┐                          │
  │          │                     │      B      │   B ∈ R^{d_out × r}     │
  │          │                     │  (TRAIN)    │   Zero init              │
  │          │                     │  d_out × r  │                          │
  │          │                     └──────┬──────┘                          │
  │          │                            │                                  │
  │          │                            ▼                                  │
  │          │                      × (α / r)    scaling                     │
  │          │                            │                                  │
  │          └────────────┬───────────────┘                                  │
  │                       ▼                                                  │
  │                  h = W₀x + (α/r)·BAx   ∈ R^{d_out}                      │
  │                                                                          │
  │   Trainable params:  r·d_in + r·d_out = r(d_in + d_out)               │
  │   Frozen params:     d_in · d_out                                        │
  └─────────────────────────────────────────────────────────────────────────┘

  Multi-head attention placement (one query projection shown):

       x ──► [W_Q + B_Q A_Q] ──► Q ──┐
       x ──► [W_K  (frozen)  ] ──► K ──┼──► softmax(QK^T/√d_k) V
       x ──► [W_V + B_V A_V] ──► V ──┘
```

---

### 6. Parameter Count

For a square layer ($d_{\text{in}} = d_{\text{out}} = d$):

| Component | Count | Example ($d=768, r=8$) |
|-----------|-------|------------------------|
| Full weight $W_0$ | $d^2$ | 589,824 |
| LoRA matrices $B, A$ | $2dr$ | 12,288 |
| **Fraction trainable** | $\dfrac{2dr}{d^2} = \dfrac{2r}{d}$ | 2.1% |
| **Savings** | $1 - \dfrac{2r}{d}$ | **97.9%** |

General savings formula:

$$\text{Savings} = 1 - \frac{2r}{d} = 1 - \frac{2 \times 8}{768} = 0.979 \;\;(97.9\%)$$

For a transformer with $L$ layers and LoRA applied to $k$ projection matrices per layer:

$$\text{Total LoRA params} = L \cdot k \cdot 2dr$$

---

### 7. Which Layers to Apply LoRA To?

| Model | Recommended targets | Rationale |
|-------|---------------------|-----------|
| **CLIP** (ViT + text encoder) | $W_Q$, $W_V$ only | Empirically sufficient; halves adapter params vs. all four projections |
| **LLaMA / Mistral** | $W_Q, W_K, W_V, W_O$ (all attention) | Larger capacity needed for generative tasks |
| **MLP / FFN** | Optional second stage | Diminishing returns; attention layers carry most task signal |

**Why $Q$ and $V$ but not $K$?** (Meng et al., 2024; Hu et al., 2021)

Attention output: $\text{softmax}(QK^\top / \sqrt{d_k}) V$

- **Query ($Q$):** Controls *which* tokens the model attends to — directly shapes the attention distribution. Adapting $W_Q$ retargets attention patterns for the downstream task.
- **Value ($V$):** Controls *what information* is retrieved and passed forward. Adapting $W_V$ changes the semantic content written to the residual stream.
- **Key ($K$):** Defines the *address* that queries match against. $Q$ and $K$ enter symmetrically in $QK^\top$, but adapting both is **redundant** in a low-rank setting: $\Delta(QK^\top) \approx (\Delta Q) K^\top + Q (\Delta K)^\top$ can be approximated by adapting $Q$ alone when $K$ is frozen (the frozen $K$ acts as a fixed addressing scheme). Empirically, $Q + V$ matches $Q + K + V$ quality at half the parameter cost.

```
  Attention head (CLIP-style LoRA):

       Q-path:  x ──► W_Q (frozen) ──► Q ──┐
                    └── B_Q A_Q (LoRA) ──┘   │
                                             ├──► Attn(Q,K,V) ──► output
       K-path:  x ──► W_K (frozen) ──► K ────┤
                                             │
       V-path:  x ──► W_V (frozen) ──► V ──┐
                    └── B_V A_V (LoRA) ──┘   │
```

---

### 8. LoRA Merging — Zero Additional Inference Cost

After training, fold the adapter into the base weights:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

At inference:

$$h = W_{\text{merged}} \, x = W_0 x + \frac{\alpha}{r} BAx$$

**One matrix multiply. No adapter overhead.** This is a key advantage over bottleneck adapters (which always add latency) and prefix tuning (which lengthens the KV cache).

```
  Training:                          Inference (after merge):

  x ──► W₀x ──┐                      x ──► W_merged x ──► h
        BAx ──┴──► h                 (single matmul, same FLOPs as base model)
```

Merge is exact (no approximation error) because matrix addition is exact in fp16/fp32.

---

### 9. Rank Selection Guide

| Rank $r$ | Params ($d=768$) | Use Case | Expected Quality |
|----------|------------------|----------|------------------|
| 1–2 | 1.5K–3K | Probing / sanity checks | Baseline |
| 4 | 6K | Simple classification, sentiment | Good |
| **8** | **12K** | **Default starting point** | **Great** |
| 16 | 25K | Domain shift, style transfer | Better |
| 32 | 49K | Major distribution shift | Very good |
| 64 | 98K | Near full-rank for $d=768$ | Best (diminishing returns) |
| 128+ | 196K+ | When $d_{\text{int}}$ is large | Approaches full FT cost |

Rule of thumb: start at $r=8$, double if validation loss plateaus above full-FT baseline, halve if overfitting on small data.

---

## Notebook 2: `02_qlora_4bit_finetuning.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/02_qlora_4bit_finetuning.ipynb)

QLoRA (Dettmers et al., 2023) combines **4-bit weight quantization** of the frozen base model with **fp16 LoRA adapters**. The base model shrinks by 8×; only the tiny adapter matrices are trained in full precision.

---

### 1. Quantization Mathematics

#### Affine (Asymmetric) Quantization

Map a floating-point tensor to $b$-bit integers:

$$q = \text{round}\!\left(\frac{x - z}{s}\right), \qquad s = \frac{x_{\max} - x_{\min}}{2^b - 1}$$

where $s$ is the **scale**, $z$ is the **zero-point** (maps real zero to an integer), and $q \in \{0, 1, \ldots, 2^b - 1\}$.

Dequantization reconstructs:

$$\hat{x} = s \cdot q + z$$

Quantization error per element:

$$\epsilon = x - \hat{x}, \qquad \|\epsilon\|_\infty \leq \frac{s}{2}$$

#### Symmetric Quantization

When the distribution is symmetric about zero (typical for neural network weights), set $z = 0$:

$$s = \frac{|x|_{\max}}{2^{b-1} - 1}, \qquad q = \text{round}\!\left(\frac{x}{s}\right), \qquad \hat{x} = s \cdot q$$

Symmetric quantization uses the full $b$-bit range for magnitude and avoids zero-point overhead, but wastes levels if $x_{\min} \neq -x_{\max}$.

---

### 2. NF4 — Normal Float 4-bit

Uniform 4-bit quantization places levels at equal intervals — wasteful for $\mathcal{N}(0, \sigma^2)$-distributed weights where most mass concentrates near zero.

**NF4** (Normal Float 4) places levels at the **quantiles** of the standard normal:

$$q_i = \Phi^{-1}\!\left(\frac{2i + 1}{2 \cdot 2^b}\right), \quad i = 0, 1, \ldots, 2^b - 1$$

where $\Phi^{-1}$ is the inverse CDF of $\mathcal{N}(0,1)$.

#### The 16 NF4 Values

$$\{-1.0,\; -0.6962,\; -0.5251,\; -0.3949,\; -0.2844,\; -0.1848,\; -0.0911,\; 0,\; 0.0796,\; 0.1609,\; 0.2461,\; 0.3379,\; 0.4407,\; 0.5626,\; 0.7230,\; 1.0\}$$

(16 levels for $b=4$; normalized to $[-1, 1]$ after block-wise scaling.)

#### Weight Distribution with NF4 Levels Overlaid

```
  p(w)
   │
   │            ╭──╮
   │           ╱    ╲
   │          ╱      ╲
   │         ╱        ╲
   │        ╱          ╲
   │───────╱────────────╲──────────────────► w
        -1.0          0          1.0

  NF4 levels (|  = quantization bin):
  |  |   |  | | | | | | | | | |  |   |  |
  -1 -.70 -.53 -.39 -.28 -.18 -.09 0 .08 .16 .25 .34 .44 .56 .72 1.0
       ▲                           ▲
   sparse bins                  dense bins
   (few weights)               (many weights)
```

Each weight block is independently scaled: $w \approx s \cdot \text{NF4}(w/s)$ where $s$ is the block's absmax. NF4 minimizes expected $|w - \hat{w}|^2$ under the Gaussian assumption.

---

### 3. Double Quantization

Each NF4 block stores a fp32 scale constant $c_1$ (one per block of, e.g., 64 weights). For a 7B model with block size 64:

$$\text{\# blocks} = \frac{7 \times 10^9}{64} \approx 1.09 \times 10^8, \qquad \text{scale storage} = 1.09 \times 10^8 \times 4 \text{ bytes} \approx 0.44 \text{ GB}$$

**Double quantization** compresses the scale constants themselves:

$$c_2 = \text{quant}_{\text{fp8}}(c_1), \qquad \hat{c}_1 = \text{dequant}(c_2)$$

This reduces scale storage from fp32 to fp8, saving approximately **0.37 bits per parameter** across the full model — enough to fit a 7B model in ~3.5 GB instead of ~3.9 GB.

```
  Weight w  ──►  NF4 quant (scale c₁)  ──►  4-bit storage
                      │
                      ▼
                 c₁ (fp32)  ──►  fp8 quant  ──►  c₂ (8-bit storage)
                                    │
                                    ▼
                              dequant(c₂) ≈ c₁  at runtime
```

---

### 4. Paged Optimizers

Adam stores $m, v \in \mathbb{R}^{|\theta|}$ for every trainable parameter. Even with LoRA, peak memory spikes during the optimizer step when gradients, parameters, and states coexist.

**Paged optimizers** (bitsandbytes) treat GPU memory as a cache:

```
  GPU VRAM (hot)                    CPU RAM (cold / page pool)
  ┌──────────────────┐              ┌──────────────────────────┐
  │ LoRA params (A,B)│              │ Optimizer states m, v    │
  │ Active gradients │  ◄─page──►  │ (paged in/out on demand) │
  │ NF4 base weights │              │ Overflow buffer          │
  └──────────────────┘              └──────────────────────────┘

  On OOM during optimizer.step():  spill m, v pages to CPU
  On next step:                    page back only needed shards
```

This prevents training crashes from transient memory spikes without changing the mathematical update rules.

---

### 5. QLoRA Memory Breakdown — 7B Model

| Component | Precision | Size Formula | Memory |
|-----------|-----------|--------------|--------|
| Base model weights | NF4 (4-bit) | $7\text{B} \times 0.5$ B | **3.50 GB** |
| Quantization scales | fp8 (double quant) | $\approx 7\text{B} \times 0.037$ B | **0.26 GB** |
| LoRA adapters ($A, B$) | fp16 | $2 \times L \times k \times 2dr \times 2$ B | **~16 MB** |
| LoRA gradients | fp16 | same as adapters | **~16 MB** |
| LoRA optimizer ($m, v$) | fp32 | $4 \times |\theta_{\text{LoRA}}|$ | **~32 MB** |
| Activations (seq=512) | fp16 | batch × seq × hidden × layers | **~0.5 GB** |
| CUDA / framework overhead | — | fixed | **~0.5 GB** |
| **Total (typical)** | | | **~4.0 GB** |

Compare: full fp32 finetuning = **112 GB** → QLoRA = **~4 GB** → **28× reduction**.

---

### 6. Memory Comparison (ASCII Bar Chart)

```
  GPU Memory Required — 7B Parameter Model
  (each █ ≈ 4 GB)

  Full FT (fp32)  ████████████████████████████  112 GB
  Full FT (fp16)  ██████████████                 56 GB
  LoRA only (fp16) ████                           14 GB
  LoRA + INT8 base ███                             7 GB
  QLoRA (NF4)      ██                              4 GB  ◄── Colab T4 (15 GB) ✓

                  |    |    |    |    |    |
                  0   20   40   60   80  100  120 GB
```

---

### 7. QLoRA Forward Pass

$$h = \underbrace{\text{Dequant}(W_{\text{NF4}})}_{\text{frozen, computed on-the-fly}} x + \frac{\alpha}{r} \underbrace{BA}_{\text{trainable, fp16}} x$$

Gradients flow **only** through $A$ and $B$. The NF4 base weights are dequantized to fp16/bf16 at compute time, then discarded — never stored in fp16 on GPU.

---

## Notebook 3: `03_adapter_methods.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/03_adapter_methods.ipynb)

Six PEFT methods, each with full mathematical specification, parameter accounting, architecture diagram, and trade-offs.

---

### Method 1: LoRA (Low-Rank Adaptation)

**Formulation.** For $W_0 \in \mathbb{R}^{d \times d}$, $x \in \mathbb{R}^d$:

$$h = W_0 x + \frac{\alpha}{r} BAx, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times d}$$

**Parameter count:** $\boxed{2dr}$ per adapted layer.

```
  x ──► W₀ (frozen) ──────────────────────┐
  x ──► A (r×d) ──► B (d×r) ──×(α/r) ────┴──► h
        └──── trainable: 2dr params ────┘
```

| Pros | Cons |
|------|------|
| Mergeable → zero inference overhead | Rank $r$ must be tuned |
| Strong empirical results across tasks | Only linear layers (need extension for LayerNorm) |
| No change to sequence length | $Q$-only or $Q,V$ selection requires domain knowledge |

---

### Method 2: Bottleneck Adapter (Houlsby et al., 2019)

**Formulation.** Inserted after attention and/or FFN sublayers:

$$h = x + f(x), \qquad f(x) = W_{\text{up}} \cdot \text{ReLU}(W_{\text{down}} x + b_{\text{down}}) + b_{\text{up}}$$

where $W_{\text{down}} \in \mathbb{R}^{r \times d}$, $W_{\text{up}} \in \mathbb{R}^{d \times r}$, $b_{\text{down}} \in \mathbb{R}^r$, $b_{\text{up}} \in \mathbb{R}^d$.

**Parameter count:** $\boxed{2dr + d + r}$ per adapter module.

```
  x ──────────────────────────────────────────────┐
  │                                                │
  ▼                                                │
  W_down (r×d) ──► ReLU ──► W_up (d×r) ────────────┴──► h = x + f(x)
  └── bottleneck r << d ──┘
```

| Pros | Cons |
|------|------|
| Nonlinear adaptation (ReLU) | **Always adds latency** (cannot merge) |
| Modular — swap adapters per task | Two forward passes through adapter per layer |
| Works well for multi-task serving | More params than LoRA at same $r$ |

---

### Method 3: Prefix Tuning (Li & Liang, 2021)

**Formulation.** Prepend $n$ learnable prefix vectors to keys and values in **every** attention layer:

$$K' = [P_K;\; K], \quad V' = [P_V;\; V], \quad P_K, P_V \in \mathbb{R}^{n \times d_k}$$

$$\text{Attn}(Q, K', V') = \text{softmax}\!\left(\frac{Q {K'}^\top}{\sqrt{d_k}}\right) V'$$

where $[;]$ denotes concatenation along the sequence dimension. A small MLP can generate $P_K, P_V$ from a shared prefix embedding to reduce params.

**Parameter count (direct):** $\boxed{2n \cdot d_k}$ per layer. With MLP reparameterization: $\boxed{n \cdot d + 2L \cdot n \cdot d_k}$ total.

```
  Learnable prefix P_K, P_V (n tokens)
         │
         ▼
  K = [P_K ; K_real]     Q ──► QK'^T / √d_k ──► softmax ──► × V' ──► out
  V = [P_V ; V_real]
       ▲
  n virtual tokens prepended to every layer's KV cache
```

| Pros | Cons |
|------|------|
| No modification to pretrained weights | Increases KV cache length by $n$ at every layer |
| Strong for generation tasks | Inference memory grows with $n \times L$ |
| Task-specific "soft instructions" | Weaker on classification vs. LoRA |

---

### Method 4: Prompt Tuning (Lester et al., 2021)

**Formulation.** Prepend $n$ learnable soft prompt embeddings **only at the input** (not per-layer):

$$\mathbf{Z}_0 = [P;\; E(x)] \in \mathbb{R}^{(n+T) \times d}, \quad P \in \mathbb{R}^{n \times d}$$

The transformer processes $[P; E(x)]$ as a single sequence; only $P$ receives gradients.

**Parameter count:** $\boxed{n \cdot d}$ total (not per layer).

```
  Input tokens:  [w1, w2, w3, ..., wT]
                  ▼  ▼  ▼       ▼
  Embeddings:   [e1, e2, e3, ..., eT]

  Prompt tuning:
  [p1, p2, ..., pn | e1, e2, ..., eT]  ──► Transformer (all frozen) ──► logits
   └─ trainable ─┘   └── frozen ──┘
```

| Pros | Cons |
|------|------|
| Extremely few parameters ($n \cdot d$) | Needs large base models ($>1$B) to work well |
| Single insertion point | No per-layer control |
| Easy multi-task (swap prompt vectors) | Sensitive to prompt length $n$ |

---

### Method 5: IA3 (Infused Adapter, Liu et al., 2022)

**Formulation.** Learn three rescaling vectors per layer that element-wise multiply activations:

$$K' = \ell_K \odot K, \quad V' = \ell_V \odot V, \quad \text{FFN}'(x) = \ell_{\text{ff}} \odot \text{FFN}(x)$$

where $\ell_K, \ell_V, \ell_{\text{ff}} \in \mathbb{R}^d$ are initialized to $\mathbf{1}$ (identity at start).

**Parameter count:** $\boxed{3d}$ per layer.

```
  K ──► (× ℓ_K) ──► K'  ──┐
  V ──► (× ℓ_V) ──► V'  ──┼──► Attention
  FFN(x) ──► (× ℓ_ff) ──► FFN'(x)

  ℓ_K, ℓ_V, ℓ_ff ∈ R^d  (3d params per layer, element-wise only)
```

| Pros | Cons |
|------|------|
| Minimal parameters (3d per layer) | Linear scaling only — no rank-$r$ expressivity |
| No architectural change | Weaker than LoRA on hard tasks |
| Fast to train | Cannot merge (runtime multiply) |

---

### Method 6: BitFit (Zaken et al., 2021)

**Formulation.** Freeze all weight matrices; train only bias vectors:

$$\theta_{\text{train}} = \{b_1, b_2, \ldots, b_L\}, \qquad \theta_{\text{frozen}} = \{W_1, W_2, \ldots, W_L\}$$

Each layer's forward pass: $h = Wx + b$ with $W$ frozen, $b$ trainable.

**Parameter count:** $\boxed{d}$ per layer (one bias vector per linear layer).

```
  x ──► W (frozen, d×d) ──► Wx ──► (+ b trainable) ──► h
                                      ▲
                                   d params
```

| Pros | Cons |
|------|------|
| Simplest possible adaptation | Limited capacity — biases shift output, not representations |
| Trivial to implement | Poor on generative / complex tasks |
| ~0.1% of full params | No merge benefit (but overhead is negligible) |

---

### Grand Comparison Table

| Method | Formula | Params ($d=768$) | % of Full ($d^2$) | Pros | Cons |
|--------|---------|------------------|-------------------|------|------|
| **Full FT** | all $\theta$ | 589,824 | 100% | Maximum capacity | 112 GB for 7B |
| **Adapter** ($r=64$) | $x + W_{\text{up}}\,\text{ReLU}(W_{\text{down}} x)$ | 99,136 | 16.8% | Nonlinear, modular | Inference latency |
| **Prefix** ($n=10$) | $[P_K;K], [P_V;V]$ | 15,360 | 2.6% | Good for generation | KV cache bloat |
| **Prompt** ($n=20$) | $[P; E(x)]$ | 15,360 | 2.6% | Minimal, multi-task | Needs large models |
| **LoRA** ($r=8$) | $W_0 x + \frac{\alpha}{r}BAx$ | 12,288 | **2.1%** | Mergeable, strong | Rank tuning |
| **IA3** | $\ell \odot K,\; \ell \odot V$ | 2,304 | 0.4% | Ultra-light | Limited expressivity |
| **BitFit** | train $\{b_i\}$ only | 768 | **0.13%** | Trivial | Weakest capacity |

---

## Notebook 4: `04_finetune_clip_custom_data.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/04_finetune_clip_custom_data.ipynb)

End-to-end LoRA finetuning of CLIP on custom image–text pairs: from pretrained checkpoint to merged deployment weights.

---

### The 4-Step Pipeline

```
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  Step 1     │    │  Step 2     │    │  Step 3     │    │  Step 4     │
  │  PRETRAIN   │───►│ FREEZE+LoRA │───►│  FINETUNE   │───►│   MERGE     │
  │  (done)     │    │  inject A,B │    │  train A,B  │    │  W=W₀+BA    │
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Step 1 — Pretrain base CLIP** (already done by OpenAI):

$$\min_{\theta} \mathcal{L}_{\text{CLIP}}(\theta) = -\sum_{i} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_j \exp(\text{sim}(I_i, T_j) / \tau)}$$

All $\theta$ trainable on 400M image–text pairs.

**Step 2 — Freeze + inject LoRA:**

$$\theta = \{\underbrace{W_0}_{\text{frozen}},\; \underbrace{A, B}_{\text{trainable}}\}, \quad \Delta W = \mathbf{0} \text{ at init}$$

Apply to $W_Q, W_V$ in both image and text encoders.

**Step 3 — Finetune LoRA only on custom data:**

$$\min_{A, B} \mathcal{L}_{\text{CLIP}}\!\left(W_0 + \tfrac{\alpha}{r} BA,\; \mathcal{D}_{\text{custom}}\right)$$

Only $A, B$ receive gradients. Memory: ~2 GB for ViT-B/32 + LoRA.

**Step 4 — Merge at inference:**

$$W_{\text{final}} = W_0 + \frac{\alpha}{r} BA \quad \Longrightarrow \quad \text{zero extra latency}$$

---

### Before vs After — Image–Text Similarity Matrix

Five custom classes (rows = images, cols = text). Diagonal = correct pairs.

```
  BEFORE LoRA (pretrained, weak on domain):     AFTER LoRA (domain-adapted):

  img\text   t0   t1   t2   t3   t4              img\text   t0   t1   t2   t3   t4
  ─────────────────────────────                  ─────────────────────────────
  i0        [.60 .30 .40 .20 .50]                i0        [.92 .05 .02 .01 .00]
  i1        [.30 .55 .35 .40 .25]                i1        [.04 .91 .03 .01 .01]
  i2        [.40 .25 .50 .30 .35]                i2        [.02 .03 .89 .04 .02]
  i3        [.20 .40 .30 .58 .42]                i3        [.01 .01 .05 .90 .03]
  i4        [.50 .20 .35 .38 .52]                i4        [.00 .02 .01 .02 .95]

  Off-diagonal ≈ 0.3–0.5 (confused)             Off-diagonal ≈ 0.0–0.05 (sharp)
  Diagonal not dominant                          Diagonal ≈ 0.9 (correct retrieval)
```

The contrastive loss $\mathcal{L}_{\text{CLIP}}$ directly optimizes these dot products; LoRA shifts $W_Q, W_V$ so that matched pairs align and unmatched pairs repel.

---

### Loss Landscape Insight — LoRA as Subspace Constrained Optimization

Full finetuning searches over $\mathbb{R}^{d^2}$. LoRA restricts the search to a **fixed-rank manifold**:

$$\mathcal{M}_r = \left\{ W_0 + BA \;\middle|\; B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times d} \right\}, \quad \dim(\mathcal{M}_r) = 2dr - r^2$$

For $d=768, r=8$: full space dimension $589{,}824$ vs. LoRA manifold dimension $\approx 12{,}224$.

```
  Loss L(W) over weight space (schematic):

  Full FT searches everywhere:          LoRA searches a thin slice:

       ╱╲                                      ╱╲
      ╱  ╲                                    ╱  ╲
     ╱    ╲                                  ╱    ╲
    ╱  *   ╲   ← many local minima         ╱  |   ╲
   ╱        ╲                            ╱   |    ╲  ← constrained path
  ╱__________╲                          ╱____|_____╲     along M_r

  High-dimensional, expensive             Low-rank subspace contains
  112 GB memory                           the task-relevant minimum
                                          (Aghajanyan: d_int << D)
```

Because $d_{\text{int}} \ll D$, the global minimum of $\mathcal{L}$ over $\mathbb{R}^{d^2}$ lies near $\mathcal{M}_r$ for small $r$ — LoRA finds it with a fraction of the memory and compute.

---

## PEFT Decision Flowchart

```
                              ┌──────────────┐
                              │  Start Here  │
                              └──────┬───────┘
                                     │
                                     ▼
                         Is your model > 1B params?
                          ╱                        ╲
                       YES                          NO
                        │                            │
                        ▼                            ▼
              Do you have < 8 GB GPU?         Is it classification?
               ╱                    ╲          ╱              ╲
            YES                      NO       YES              NO
             │                        │        │                │
             ▼                        ▼        ▼                ▼
          ┌──────┐               ┌──────┐  ┌─────────┐     ┌──────┐
          │QLoRA │               │ LoRA │  │ Prompt  │     │ LoRA │
          │NF4   │               │r=8-16│  │ or IA3  │     │ r=8  │
          │r=8   │               └──────┘  └─────────┘     └──────┘
          └──────┘               ~14 GB      ~minimal        ~low
          ~4 GB

                                     │
                                     ▼
                         Need zero inference overhead?
                          ╱                        ╲
                       YES                          NO
                        │                            │
                        ▼                            ▼
                   ┌─────────┐               ┌──────────────┐
                   │  LoRA   │               │ Adapter or   │
                   │ (merge) │               │ Prefix Tuning│
                   └─────────┘               └──────────────┘

                                     │
                                     ▼
                         Multi-task serving?
                          ╱                        ╲
                       YES                          NO
                        │                            │
                        ▼                            ▼
              ┌──────────────────┐            ┌─────────────┐
              │ Adapter / Prompt │            │ LoRA / QLoRA│
              │ (swap modules)   │            │ (single task)│
              └──────────────────┘            └─────────────┘
```

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total notebooks | 4 |
| Total visualizations | 14 |
| PEFT methods covered | 6 (LoRA, Adapter, Prefix, Prompt, IA3, BitFit) |
| Key equation | $h = W_0 x + \frac{\alpha}{r} BAx$ |
| 7B full FT memory | 112 GB |
| 7B QLoRA memory | ~4 GB |
| Parameter savings (LoRA, $r=8$, $d=768$) | 97.9% |
| Runs on CPU | Yes (slow); GPU recommended |

---

## Next Step

**[05_Advanced_Topics/01_llava_architecture.ipynb](../05_Advanced_Topics/01_llava_architecture.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/01_llava_architecture.ipynb) — how LLaVA connects vision to LLMs.
