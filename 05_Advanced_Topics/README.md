# Module 05: Advanced Topics

> **Time:** 2-3 hours | **Notebooks:** 3 | **Visuals:** 7 plots | **Difficulty:** Advanced

| Notebook | Open in Colab |
|----------|---------------|
| `01_llava_architecture.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/01_llava_architecture.ipynb) |
| `02_multimodal_beyond_vision.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/02_multimodal_beyond_vision.ipynb) |
| `03_efficient_deployment.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/03_efficient_deployment.ipynb) |

---

## Journey Recap — Module 00 → 05

```
  Module 00        Module 01         Module 02          Module 03         Module 04         Module 05
  =========        =========         =========          =========         =========         =========
  Environment      Encoders &        CLIP, VQA,         Training          LoRA / QLoRA      YOU ARE
  & Setup          Fusion            Captioning         Pipeline          PEFT              HERE -->
       |                |                  |                  |                 |                 |
       v                v                  v                  v                 v                 v
  +---------+    +-----------+      +-----------+      +-----------+     +-----------+     +-----------+
  | PyTorch |    | ViT, CNN  |      | Contrastive|      | Loss fns  |     | Low-rank  |     | LLaVA     |
  | GPU     |    | Cross-    |      | InfoNCE    |      | Optimizers|     | adapters  |     | Audio/    |
  | setup   |    | attention |      | Zero-shot  |      | Schedulers|     | 4-bit Q   |     | Video     |
  +---------+    +-----------+      +-----------+      +-----------+     +-----------+     | ONNX/INT8 |
                                                                                              +-----------+

  "I set up"     "I understand      "I can build       "I can train      "I can finetune   "I can build
                  multimodal          CLIP-style          end-to-end        on limited        SOTA MLLMs
                  encoders"           models"             pipelines"        hardware"         & deploy them"

  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  Mathematical arc:  dot products → softmax attention → contrastive loss → autograd → low-rank
                       factorization → causal LM + KV cache → STFT/Mel → quantization error bounds
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## Notebook 1: `01_llava_architecture.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/01_llava_architecture.ipynb)

### LLaVA — The Mathematics of Multimodal LLMs

LLaVA's core insight: **project visual patch embeddings into the LLM token space** and treat them as ordinary sequence positions. The LLM performs causal self-attention over visual and textual tokens jointly — no cross-attention module required.

```
  Data flow (tensor shapes at each stage):

  Image I  ──► CLIP-ViT ──► Z_img [576 × 1024]
                                    │
                                    ▼
                              MLP Projector
                                    │
                                    ▼
                              H_img [576 × 4096] ──┐
                                                   ├──► X [576+T × 4096] ──► LLM ──► logits
  Text prompt ──► Tokenizer + Embed ──► E [T × 4096] ──┘
```

### 1. CLIP ViT Feature Extraction

Given input image $I \in \mathbb{R}^{3 \times H \times W}$, the CLIP Vision Transformer produces a sequence of patch token embeddings:

$$\mathbf{Z}_{\text{img}} = \text{CLIP-ViT}(I) \in \mathbb{R}^{N_p \times D_v}$$

**Patch count derivation** (ViT-L/14 @ 336px):

$$N_{\text{side}} = \frac{H_{\text{img}}}{P} = \frac{336}{14} = 24, \qquad N_p = N_{\text{side}}^2 = 24 \times 24 = 576$$

Each patch covers a $14 \times 14$ pixel region. With $D_v = 1024$ (ViT-L hidden dim):

$$\mathbf{Z}_{\text{img}} \in \mathbb{R}^{576 \times 1024}$$

Internally, ViT applies $L_v = 24$ transformer layers with multi-head self-attention:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

The ViT is **frozen** during LLaVA training — only the projector (and optionally LoRA weights) are updated.

### 2. MLP Projector — Bridging Modalities

A 2-layer MLP maps each visual token from $D_v$ to the LLM hidden dimension $D_t$:

$$\mathbf{H} = W_2 \cdot \text{GELU}(W_1 \mathbf{Z}_{\text{img}} + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{D_v \times D_h}$, $W_2 \in \mathbb{R}^{D_h \times D_t}$, and GELU is:

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right)$$

**Dimension flow:**

$$\mathbb{R}^{576 \times 1024} \xrightarrow{W_1} \mathbb{R}^{576 \times 4096} \xrightarrow{\text{GELU}} \mathbb{R}^{576 \times 4096} \xrightarrow{W_2} \mathbb{R}^{576 \times 4096}$$

**Parameter count derivation** ($D_v = 1024$, $D_h = D_t = 4096$):

$$|\theta_{\text{proj}}| = \underbrace{D_v D_h}_{W_1} + \underbrace{D_h}_{b_1} + \underbrace{D_h D_t}_{W_2} + \underbrace{D_t}_{b_2}$$

$$= 1024 \times 4096 + 4096 + 4096 \times 4096 + 4096 = 4{,}198{,}528 + 16{,}781{,}312 + 4096 \approx \mathbf{21\text{M}}$$

This is $\ll 1\%$ of the 7B LLM — yet it is the **only** learned interface between vision and language.

**Why MLP instead of a single linear layer?**

A linear map $W \mathbf{z}$ can only perform a change of basis. The modality gap between CLIP's contrastive space and the LLM's next-token prediction space is **nonlinear**. GELU introduces curvature that lets the projector learn compositional transforms (e.g., "combine color + shape cues into a linguistic concept vector") that a linear map cannot represent:

```
  Linear projector:          MLP projector (2-layer):
  z ──► W·z ──► LLM space    z ──► W₁·z ──► GELU ──► W₂·(·) ──► LLM space
       (rotation +              (rotation + nonlinear
        scaling only)            warp + re-rotation)
```

### 3. Sequence Construction

Visual and text embeddings are concatenated along the sequence dimension:

$$\mathbf{X} = \left[\mathbf{H}_{\text{img}}^1, \ldots, \mathbf{H}_{\text{img}}^{576},\; \mathbf{e}_1, \ldots, \mathbf{e}_T\right] \in \mathbb{R}^{(576 + T) \times D_t}$$

Position $i \in \{1, \ldots, 576\}$ holds a visual token; positions $577, \ldots, 576+T$ hold text tokens. The LLM's learned positional embeddings are applied to **all** positions uniformly — the model discovers that early positions carry visual semantics.

```
  Sequence layout (causal attention sees this ordering):

  Index:  [  1  |  2  | ... | 576 | 577 | 578 | ... | 576+T ]
  Token:  [ H¹  | H²  | ... | H⁵⁷⁶| e₁  | e₂  | ... |  e_T  ]
  Type:   [-------- visual (576) --------|-- text (T) --]

  Text token e_j can attend to:  all 576 visual tokens + e_1, ..., e_{j-1}
  Visual token H^i can attend to: H^1, ..., H^i  (causal among visuals too)
```

### 4. Causal Attention with Visual Tokens

The LLM applies **causal (masked) self-attention** over all $N = 576 + T$ tokens in each of $L$ layers:

$$\text{Attn}(\mathbf{X}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}, \qquad \mathbf{M}_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

**Attention complexity per layer:**

$$\mathcal{O}(N^2 \cdot d) = \mathcal{O}\!\left((576 + T)^2 \cdot d\right)$$

For $T = 128$ and $d = 4096$: each layer performs $\approx (704)^2 \times 4096 \approx 2.0 \times 10^9$ multiply-adds. The 576 visual tokens dominate the quadratic cost.

**KV Cache during autoregressive generation:**

At inference, recomputing keys and values for all past tokens at every step is wasteful. Instead, cache $\mathbf{K}, \mathbf{V}$ from prior steps:

$$\text{Attn}(\mathbf{q}_{\text{new}}, [\mathbf{K}_{\text{cached}}; \mathbf{k}_{\text{new}}], [\mathbf{V}_{\text{cached}}; \mathbf{v}_{\text{new}}])$$

**Memory footprint** (per layer, per sequence):

$$\text{Mem}_{\text{KV}} = 2 \times L \times (576 + T) \times d \times \text{bytes}$$

The factor of 2 accounts for separate K and V tensors. For FP16 with $L = 32$, $T = 512$, $d = 4096$:

$$\text{Mem}_{\text{KV}} = 2 \times 32 \times 1088 \times 4096 \times 2 \text{ bytes} \approx 576 \text{ MB}$$

Each new generated token computes $\mathbf{q}_{\text{new}} \mathbf{K}_{\text{cached}}^\top$ — attention against **all** cached keys — then appends its own $(\mathbf{k}_{\text{new}}, \mathbf{v}_{\text{new}})$ to the cache.

```
  KV Cache growth during generation:

  Step 0 (prefill):  cache ← all 576 visual + prompt tokens
  Step 1:            q₁ · [K_cache] → append (k₁, v₁)
  Step 2:            q₂ · [K_cache, k₁] → append (k₂, v₂)
  ...
  Step t:            q_t · [K_cache, k₁, ..., k_{t-1}] → append (k_t, v_t)

  Memory grows linearly in t; compute per step also grows linearly in t.
```

### 5. Two-Stage Training Protocol

**Stage 1 — Projector alignment** (freeze ViT + LLM):

$$\min_{\theta_{\text{proj}}} \mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T'} \log P\!\left(y_t \mid y_{<t}, I;\; \theta_{\text{ViT}}^*,\; \theta_{\text{proj}},\; \theta_{\text{LLM}}^*\right)$$

Only $\theta_{\text{proj}}$ receives gradients. Data: **595K** image-caption pairs (LAION/CC/SBU). Cost: ~5 h on 8×A100.

**Stage 2 — Instruction tuning** (projector + LoRA on LLM):

$$\min_{\theta_{\text{proj}},\, A,\, B} \mathcal{L}_{\text{LM}} = -\sum_{t} \log P\!\left(y_t \mid y_{<t}, I;\; \theta_{\text{ViT}}^*,\; \theta_{\text{proj}},\; \theta_{\text{LLM}}^* + BA\right)$$

LoRA injects low-rank updates: $W' = W + BA$ where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, $r \ll d$. Data: **150K** instruction-following pairs (GPT-4 generated). Cost: ~20 h on 8×A100.

```
  Training schedule:

  Stage 1                    Stage 2
  ┌─────────────────┐        ┌─────────────────────────────┐
  │ ViT:  FROZEN    │        │ ViT:  FROZEN                │
  │ LLM:  FROZEN    │   ──►  │ LLM:  LoRA adapters TRAIN    │
  │ Proj: TRAIN     │        │ Proj: TRAIN                 │
  │ Data: 595K cap  │        │ Data: 150K instruct         │
  └─────────────────┘        └─────────────────────────────┘
```

### 6. LLaVA vs BLIP-2 vs GPT-4V

| Property | LLaVA | BLIP-2 | GPT-4V |
|----------|-------|--------|--------|
| **Vision encoder** | CLIP ViT-L/14 | EVA-CLIP ViT-g | Unknown (proprietary) |
| **Bridge / adapter** | 2-layer MLP projector | Q-Former (32 learnable queries) | Unknown |
| **Bridge params** | ~21M | ~188M (Q-Former) | — |
| **LLM** | Vicuna-7B/13B (LLaMA) | Flan-T5 / Vicuna | GPT-4 (RLHF) |
| **Attention pattern** | Causal self-attn over visual+text tokens | Cross-attn (Q-Former → LLM) | Unknown |
| **Training stages** | 2 (align + instruct) | 2 (Q-Former + bootstrapping) | Multi-stage RLHF |
| **Open source** | Yes (weights + code) | Yes (weights + code) | No (API only) |
| **Visual tokens** | 576 (native resolution patches) | 32 (compressed queries) | Unknown |

---

## Notebook 2: `02_multimodal_beyond_vision.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/02_multimodal_beyond_vision.ipynb)

### Extending Multimodal Learning Beyond Vision

The universal recipe: **Modality Encoder → Projection → Shared Embedding Space**. This notebook applies it to audio and video, then aligns three modalities jointly.

```
  Modality-agnostic pipeline:

  Raw signal x  ──► Feature transform  ──► Patch/tokenize  ──► Encoder  ──► z ∈ R^D
       │                    │                    │                │
   waveform            STFT / Mel           ViT-style          Transformer
   video frames        per-frame ViT        temporal attn      pooling
   text string         tokenize             —                  text encoder
```

### 1. Audio Processing — STFT Mathematics

The **Short-Time Fourier Transform** decomposes a discrete signal into time-localized frequency components:

$$X(m, \omega) = \sum_{n} x(n)\, w(n - mH)\, e^{-j \omega n}$$

where:
- $x(n)$ — discrete audio samples, sample rate $f_s$
- $w(n)$ — window function centered at frame $m$
- $H$ — hop length (samples between consecutive frames)
- $\omega = 2\pi k / N_{\text{FFT}}$ — angular frequency of bin $k$

**Hann window** (reduces spectral leakage):

$$w(n) = 0.5\left(1 - \cos\!\left(\frac{2\pi n}{N}\right)\right), \quad n = 0, \ldots, N-1$$

**Frequency resolution and Nyquist:**

$$\Delta f = \frac{f_s}{N_{\text{FFT}}}, \qquad f_{\max} = \frac{f_s}{2} \quad \text{(Nyquist limit)}$$

```
  STFT tiling of a waveform:

  x(n):  |████████████████████████████████████████|
              ↓ hop H
         [──w──]                              frame m=0
              [──w──]                         frame m=1
                   [──w──]                    frame m=2
                        ...

  Time axis →     Each column of X(m, k) is one frame's spectrum
  Freq axis ↓     Each row tracks one frequency bin over time
```

### 2. Mel Scale — Perceptual Frequency Warping

Human pitch perception is approximately logarithmic. The **Mel scale** maps Hertz to Mels:

$$m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

Inverse: $f = 700\left(10^{m/2595} - 1\right)$.

At $f = 1000$ Hz: $m \approx 1000$ Mel. The scale compresses high frequencies (where the ear is less discriminative) and expands low frequencies.

### 3. Mel Spectrogram

Apply triangular filter banks $H_f(k)$ to the power spectrum:

$$\text{MelSpec}(f, t) = \sum_{k} \left|X(k, t)\right|^2 H_f(k)$$

Each filter $H_f$ is a triangle centered at Mel frequency $f$, overlapping neighbors. The result is a 2D matrix (Mel bins × time frames) — structurally identical to a grayscale image.

$$\text{MelSpec} \in \mathbb{R}^{N_{\text{mel}} \times N_{\text frames}} \approx \mathbb{R}^{128 \times 1000}$$

### 4. Audio Patching — Spectrogram as Image

Because MelSpec is 2D, apply the ViT recipe directly:

$$\mathbf{Z}_{\text{audio}} = \text{Transformer}\!\left(\text{PatchEmbed}\!\left(\text{MelSpec}(x)\right)\right) \in \mathbb{R}^{N_p^{\text{audio}} \times D}$$

With patch size $P_a \times P_a$ on the spectrogram:

$$N_p^{\text{audio}} = \frac{N_{\text{mel}}}{P_a} \times \frac{N_{\text{frames}}}{P_a}$$

Each patch captures a local time-frequency texture (e.g., a vowel formant, a consonant burst).

```
  Mel spectrogram → patches:

  Freq (Mel)
    ▲  ┌──┬──┬──┬──┐
    │  │P1│P2│P3│..│   Each patch P_i ∈ R^{P_a × P_a}
    │  ├──┼──┼──┼──┤   Flattened + linear proj → token embedding
    │  │..│..│..│..│
    └──► Time ──────────►
```

### 5. Video Encoder — Spatial + Temporal Modeling

**Frame sampling** from a video of total length $T_{\text{total}}$ with $N_f$ source frames, selecting $F$ uniformly:

$$f_i = I_{\lfloor i \cdot T_{\text{total}} / F \rfloor}, \quad i = 0, 1, \ldots, F-1$$

**Per-frame spatial encoding:**

$$\mathbf{z}_f = \text{ViT}(I_{f_i}) \in \mathbb{R}^D, \quad f = 0, \ldots, F-1$$

**Temporal positional encoding:**

$$\mathbf{z}_f' = \mathbf{z}_f + \mathbf{e}_{\text{time}}^f$$

where $\mathbf{e}_{\text{time}}^f$ is a learned or sinusoidal encoding of frame index $f$.

**Temporal self-attention** across the $F$ frame embeddings:

$$\text{TemporalAttn}(\mathbf{Z}') = \text{softmax}\!\left(\frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}}\right)\mathbf{V}_t, \qquad \mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t \in \mathbb{R}^{F \times d}$$

Complexity: $\mathcal{O}(F^2 d)$ — typically $F = 8\text{–}32$, so this is cheap compared to spatial attention over all patches.

**Spatial-temporal factorization** (used in VideoMAE, TimeSformer):

```
  Factorized attention (avoids O(F² · N_p²)):

  Full joint:     O((F · N_p)² d)     ← intractable

  Factorized:     Spatial attn per frame:  O(F · N_p² d)
                  Temporal attn per patch:  O(N_p · F² d)

  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ Frame 0 │   │ Frame 1 │   │ Frame F │   ← spatial ViT per frame
  └────┬────┘   └────┬────┘   └────┬────┘
       └──────────┬───────────────┘
                  ▼
           Temporal attention
           (across frames)
                  ▼
           Mean pool → h_video
```

**Mean pooling** over temporally-refined frame embeddings:

$$\mathbf{h}_{\text{video}} = \frac{1}{F}\sum_{f=0}^{F-1} \mathbf{z}_f'' \in \mathbb{R}^D$$

### 6. Three-Modal Contrastive Loss

Given encoders producing $\mathbf{z}_I, \mathbf{z}_T, \mathbf{z}_A$ for image, text, and audio:

$$\mathcal{L} = \mathcal{L}_{\text{InfoNCE}}(I, T) + \mathcal{L}_{\text{InfoNCE}}(I, A) + \mathcal{L}_{\text{InfoNCE}}(T, A)$$

Each InfoNCE term (symmetric, batch size $B$):

$$\mathcal{L}_{\text{InfoNCE}}(X, Y) = -\frac{1}{2B}\sum_{i=1}^{B}\left[\log \frac{e^{\mathbf{z}_{X_i}\!\cdot\mathbf{z}_{Y_i}/\tau}}{\sum_j e^{\mathbf{z}_{X_i}\!\cdot\mathbf{z}_{Y_j}/\tau}} + \log \frac{e^{\mathbf{z}_{Y_i}\!\cdot\mathbf{z}_{X_i}/\tau}}{\sum_j e^{\mathbf{z}_{Y_j}\!\cdot\mathbf{z}_{X_j}/\tau}}\right]$$

This yields **three pairwise similarity matrices**, each trained to be diagonal-dominant:

```
  Three-Modal CLIP Similarity Matrices (idealized, batch=3):

  Image-Text         Image-Audio        Text-Audio
  +---+---+---+     +---+---+---+     +---+---+---+
  |.92|.05|.03|     |.88|.06|.06|     |.91|.04|.05|
  |.04|.89|.07|     |.07|.85|.08|     |.06|.87|.07|
  |.04|.06|.90|     |.05|.07|.88|     |.03|.05|.92|
  +---+---+---+     +---+---+---+     +---+---+---+

  "dog" + dog.jpg    dog.jpg + bark.wav   "dog" + bark.wav
  → high similarity  → high similarity    → high similarity
```

### 7. ImageBind — Image as the Anchor Modality

[ImageBind](https://arxiv.org/abs/2305.05665) extends the three-modal idea: use **image** as the universal anchor and align every other modality (text, audio, depth, IMU, thermal) into **image embedding space**:

$$\mathbf{z}_{m} = \text{Proj}_m\!\left(\text{Encoder}_m(x_m)\right) \in \mathbb{R}^D, \quad \mathcal{L} = \sum_{m \neq \text{image}} \mathcal{L}_{\text{InfoNCE}}(\text{image}, m)$$

Only image-text pairs are needed for most training signal; other modalities piggyback via their alignment to image. At inference, any modality can be "queried" against any other by routing through the shared image space:

```
  ImageBind alignment hub:

                    ┌──────────┐
         Text ─────►│          │
         Audio ────►│  Image   │────► Shared R^D
         Depth ────►│  Space   │
         IMU ──────►│ (anchor) │
         Thermal ──►│          │
                    └──────────┘

  Query: "find audio clip matching this depth map"
  → encode depth → z_depth ≈ z_image → nearest neighbor in audio bank
```

---

## Notebook 3: `03_efficient_deployment.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/03_efficient_deployment.ipynb)

### Efficient Deployment — Optimization Theory & Practice

Building the model is half the job. This notebook covers graph optimization, quantization error analysis, compilation pipelines, and deployment trade-offs.

```
  Optimization stack (apply in order of invasiveness):

  FP32 baseline  ──►  torch.compile  ──►  FP16/BF16  ──►  ONNX export  ──►  INT8 PTQ
  (no change)        (1.5-2×, free)     (2× mem)        (cross-platform)    (4× smaller)
```

### 1. ONNX Export — Static Graph Optimization

PyTorch executes **eagerly** (op-by-op). ONNX captures a **static computation graph** amenable to global optimization:

$$\text{PyTorch (dynamic)} \xrightarrow{\text{torch.onnx.export}} \text{ONNX graph} \xrightarrow{\text{ONNX Runtime}} \text{Fused kernels}$$

**Graph-level optimizations:**

| Optimization | Before | After |
|---|---|---|
| **Operator fusion** | MatMul → Add → ReLU (3 kernel launches) | FusedGemmAct (1 launch) |
| **Constant folding** | $y = x \cdot W + b$ where $W, b$ fixed | Precompute $W, b$ into single tensor |
| **CSE** | $a = f(x);$ $b = f(x)$ | $a = f(x);$ $b = a$ |

Memory planning reuses buffer regions when tensor lifetimes do not overlap, reducing peak RAM.

### 2. Post-Training Quantization (PTQ)

Map FP32 weights and activations to $b$-bit integers without retraining.

**Affine quantization:**

$$q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z,\; -2^{b-1},\; 2^{b-1} - 1\right)$$

Dequantize: $\hat{x} = s(q - z)$. For symmetric quantization ($z = 0$):

$$q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right),\; -2^{b-1},\; 2^{b-1} - 1\right)$$

**Calibration:** run $N_{\text{cal}}$ representative batches, collect per-tensor (or per-channel) min/max or MSE-optimal $s$.

| Granularity | Scale $s$ | Accuracy | Speed |
|---|---|---|---|
| **Per-tensor** | One $s$ per weight matrix | Lower | Fastest |
| **Per-channel** | One $s$ per output channel | Higher | Slightly slower |

**Quantization error analysis** (uniform quantizer, step size $s$):

$$\mathbb{E}\!\left[(\hat{x} - x)^2\right] = \frac{s^2}{12}$$

This is the variance of a uniform distribution on $[-s/2, s/2]$. Halving $s$ (doubling bit-width) reduces MSE by $4\times$.

**Model size:**

$$\frac{\text{Size}_{\text{INT8}}}{\text{Size}_{\text{FP32}}} = \frac{8}{32} = 25\%$$

### 3. Knowledge Distillation (Bonus)

Train a small **student** $f_s$ to mimic a large **teacher** $f_t$:

$$\mathcal{L}_{\text{KD}} = \underbrace{\alpha\, T^2\, \text{KL}\!\left(\text{softmax}\!\left(\frac{\mathbf{z}_s}{T}\right) \Big\| \text{softmax}\!\left(\frac{\mathbf{z}_t}{T}\right)\right)}_{\text{soft target loss}} + \underbrace{(1 - \alpha)\, \mathcal{L}_{\text{CE}}(\mathbf{z}_s, y)}_{\text{hard label loss}}$$

- **Temperature** $T > 1$ softens probability distributions, exposing dark knowledge (inter-class similarities) from the teacher.
- **$\alpha \in [0, 1]$** balances soft vs. hard supervision. Typical: $\alpha = 0.5$, $T = 4$.

The $T^2$ scaling compensates for gradient magnitude shrinkage as $T$ increases.

### 4. `torch.compile` — The PyTorch 2.0 Pipeline

```python
model = torch.compile(model, mode="reduce-overhead")
```

Three-stage compilation:

```
  Python forward()
        │
        ▼
  ┌─────────────┐
  │ TorchDynamo │  Capture FX graph via bytecode analysis (no source change)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ AOTAutograd │  Generate fused forward + backward graphs ahead of time
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ TorchInductor│  Lower to Triton (GPU) or AVX (CPU) kernels
  └──────┬──────┘
         ▼
  Cached compiled artifact → 1.5–2× speedup on subsequent calls
```

`mode="reduce-overhead"` minimizes Python dispatch cost (ideal for small batches / LLM decoding). `mode="max-autotune"` benchmarks multiple kernel variants (best for large matmuls).

### 5. Speed & Quality Comparison

| Method | Model Size | Latency (rel.) | Quality (rel.) | Hardware | Effort |
|--------|-----------|----------------|----------------|----------|--------|
| FP32 baseline | 100% | 1.0× | 100% | Any | None |
| `torch.compile` | 100% | ~0.55× | ~100% | PyTorch 2.0+ | 1 line |
| FP16 / BF16 | 50% | ~0.6× | ~99.8% | GPU w/ Tensor Cores | Minimal |
| ONNX Runtime | 100% | ~0.5× | ~100% | CPU / GPU / Edge | Moderate |
| INT8 PTQ | 25% | ~0.35× | ~97–99% | CPU INT8 / GPU | Calibration data |
| TensorRT FP16 | 50% | ~0.25× | ~99.5% | NVIDIA GPU only | High |
| KD (distilled) | ~25–50% | ~0.3× | ~95% | Any | Retraining |

*Latency = relative to FP32 baseline (lower is faster). Quality measured on downstream task accuracy.*

### 6. Deployment Decision Tree

```
                         Your trained model
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Need cross-platform   │
                    │ (mobile, C++, edge)?  │
                    └───────────┬───────────┘
                          YES   │   NO
                    ┌───────────┴───────────┐
                    ▼                         ▼
              ONNX Export              Need max GPU throughput?
              + ORT / TF Lite          /                    \
                                    YES                      NO
                                     │                        │
                                     ▼                        ▼
                               TensorRT                  torch.compile
                               (NVIDIA only)              (1 line, universal)
                                     │
                                     ▼
                               Still too slow?
                               /            \
                             YES             NO
                              │               │
                              ▼               ▼
                         INT8 PTQ         Ship it!
                         (+ calibrate)
                              │
                              ▼
                         Still too big?
                         /            \
                       YES             NO
                        │               │
                        ▼               ▼
                   Knowledge         Ship it!
                   Distillation
                   (smaller student)

  Extras:
  ┌────────────────────┬──────────────────────────────────────────────┐
  │ Web demo           │ Gradio — ~10 lines, auto UI                  │
  │ LLM serving @scale│ vLLM — PagedAttention, continuous batching   │
  │ Multi-GPU          │ torch.distributed / DeepSpeed / FSDP          │
  └────────────────────┴──────────────────────────────────────────────┘
```

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total notebooks | 3 |
| Total visualizations | 7 |
| Models built | Mini-LLaVA, Audio Encoder, Video Encoder, 3-Modal CLIP |
| Key equations covered | STFT, Mel scale, InfoNCE, KV cache, quantization error, KD loss |
| Deployment methods | 5 (ONNX, INT8, torch.compile, Gradio, vLLM) |
| Projector parameters (LLaVA) | ~21M |
| Training data (LLaVA) | 595K pretrain + 150K instruct |
| Runs on CPU | Yes |

---

## Congratulations!

You've completed **17 notebooks**, built **10+ models from scratch**, learned **13 training/finetuning techniques**, and generated **54 visualizations**.

You now understand the full mathematical stack — from STFT and Mel filter banks, through causal attention with KV caching, to quantization error bounds and knowledge distillation — needed to **build, train, finetune, and deploy** multimodal AI models, even on limited hardware.

Go build something amazing.
