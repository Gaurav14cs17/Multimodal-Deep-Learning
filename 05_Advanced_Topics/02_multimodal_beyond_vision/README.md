# 02 — Beyond Vision: Audio + Text + Video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/02_multimodal_beyond_vision/02_multimodal_beyond_vision.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

Multimodal isn't just vision + text. This notebook extends to **audio, video, and beyond**:

- Audio representations: STFT, Mel spectrograms, Whisper embeddings
- Audio-Text models (CLAP)
- Video understanding: temporal attention, spatial-temporal factorization
- **ImageBind:** one embedding space for 6 modalities
- Building a 3-modal CLIP (image + text + audio)

---

## Part 1: Audio Representations

### Audio Processing Pipeline

```
  Raw Audio Waveform (1D signal, 16 kHz)
  │  x(t) = amplitude at time t
  │
  ▼  Step 1: Framing + Windowing
  ┌────────────────────────────────────────────────────┐
  │  Split into overlapping frames:                     │
  │  Frame length: 25ms (400 samples at 16kHz)         │
  │  Hop length: 10ms (160 samples)                    │
  │  Window: Hann window w(n) = 0.5(1 - cos(2πn/N))   │
  │  → For 10s audio: (10000ms - 25ms)/10ms + 1 ≈ 998 frames │
  └────────────────────┬───────────────────────────────┘
                       │
  ▼  Step 2: STFT (Short-Time Fourier Transform)
  ┌────────────────────────────────────────────────────┐
  │  Apply FFT to each windowed frame:                  │
  │  X(k, t) = Σ x(n + tH) · w(n) · e^(-j2πkn/N)     │
  │  Output: Complex spectrogram ∈ ℂ^(201 × 998)       │
  │  (201 frequency bins × 998 time frames)            │
  └────────────────────┬───────────────────────────────┘
                       │
  ▼  Step 3: Mel Filter Bank
  ┌────────────────────────────────────────────────────┐
  │  Apply 80 triangular filters on mel scale:          │
  │  Mel spectrogram ∈ ℝ^(80 × 998)                   │
  │  + Log compression: log(mel + ε)                   │
  └────────────────────┬───────────────────────────────┘
                       │
  ▼  Step 4: Audio Transformer (e.g., Whisper encoder)
  ┌────────────────────────────────────────────────────┐
  │  Treat mel spec as "image" (80 × T):               │
  │  2 Conv1D layers → Transformer encoder             │
  │  Output: audio embedding a ∈ ℝ^d                   │
  └────────────────────────────────────────────────────┘
```

### Key Audio Equations

#### STFT (Short-Time Fourier Transform)

$$
X(k, t) = \sum_{n=0}^{N-1} x(n + tH)\, w(n)\, e^{-j2\pi kn/N}
$$

where:
- $k$ = frequency bin index ($0, 1, \ldots, N/2$)
- $t$ = time frame index
- $H$ = hop length (samples)
- $w(n)$ = window function (Hann, Hamming, etc.)
- $N$ = FFT size (typically 400 for 25ms at 16kHz)

**Power Spectrogram:**

$$
P(k, t) = \lvert X(k, t) \rvert^2
$$

#### Mel Scale — Perceptual Frequency Mapping

The mel scale maps linear frequency to perceived pitch:

$$
m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)
$$

Inverse:

$$
f = 700 \left(10^{m/2595} - 1\right)
$$

**Why mel scale?** Human hearing is approximately logarithmic — we can distinguish between 100Hz and 200Hz easily (one octave), but 10,000Hz and 10,100Hz sound nearly identical. The mel scale spaces filters accordingly.

```
  Linear scale:  |  |  |  |  |  |  |  |  |  |  | (equal Hz spacing)
  Mel scale:     |||||  |  |  |   |    |     |    (more filters at low freq)
                 0   500 1k   2k    4k    8k   16kHz
```

#### Mel Spectrogram Computation

$$
\text{MelSpec}(m, t) = \log\!\left(\sum_{k=0}^{N/2} H_m(k) \cdot \lvert X(k, t) \rvert^2 + \epsilon\right)
$$

where $H_m(k)$ is the $m$-th triangular mel filter bank.

#### Nyquist Theorem

$$
f_{\text{max}} = \frac{f_s}{2}
$$

At 16kHz sampling, we can represent frequencies up to 8kHz (covering most speech).

---

## Part 2: Audio-Text Models (CLAP)

### Architecture — CLAP (Contrastive Language-Audio Pretraining)

```
  Audio ──► [Audio Encoder] ──► a ──► [Proj] ──► â ∈ ℝᵈ ──┐
                                                              │ Cosine
                                                              │ Similarity
  Text  ──► [Text Encoder]  ──► t ──► [Proj] ──► t̂ ∈ ℝᵈ ──┘
                                                              │
                                                              ▼
                                                       InfoNCE Loss
```

**Same contrastive framework as CLIP**, but for audio-text pairs:

$$
\mathcal{L}_{\text{CLAP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{\hat{a}_i^\top \hat{t}_i/\tau}}{\sum_j e^{\hat{a}_i^\top \hat{t}_j/\tau}} + \log\frac{e^{\hat{t}_i^\top \hat{a}_i/\tau}}{\sum_j e^{\hat{t}_i^\top \hat{a}_j/\tau}}\right]
$$

---

## Part 3: Video Understanding

### Temporal Modeling — The Extra Dimension

Video adds a **time dimension** $T$ to images:

$$
\text{Image: } x \in \mathbb{R}^{H \times W \times 3}, \quad \text{Video: } X \in \mathbb{R}^{T \times H \times W \times 3}
$$

### Spatial-Temporal Factorization (TimeSFormer)

Instead of full 3D attention (which is $O(T^2 H^2 W^2)$), factorize into:

$$
\text{Attention} = \underbrace{\text{Spatial-Attn}}_{\text{within each frame}} + \underbrace{\text{Temporal-Attn}}_{\text{across frames, same position}}
$$

```
  Full Space-Time Attention:           Factorized (TimeSFormer):
  Each token attends to ALL tokens     Each token attends to:
  across ALL frames.                   1) Same frame (spatial)
                                       2) Same position, all frames (temporal)
  Cost: O(T²·N²)                      Cost: O(T·N + T²)

  Frame 1: [p₁ p₂ p₃ p₄]            Frame 1: [p₁ p₂ p₃ p₄]
  Frame 2: [p₅ p₆ p₇ p₈]                      │    │    │    │
  Frame 3: [p₉ p₁₀p₁₁p₁₂]           Frame 2: [p₅ p₆ p₇ p₈]  ←── spatial
                                                │    │    │    │
  p₁ attends to ALL 12 tokens         Frame 3: [p₉ p₁₀p₁₁p₁₂]
  (144 attention scores)                        ↑ temporal
                                       p₁ attends to p₁-p₄ (spatial)
                                       + p₁,p₅,p₉ (temporal)
                                       (7 attention scores)
```

### Spatial-Temporal Attention Equations

**Spatial attention** (per frame $t$):

$$
Z_t^{s} = Z_t + \text{MSA}_{\text{spatial}}(\text{LN}(Z_t))
$$

**Temporal attention** (across frames, per spatial position $p$):

$$
Z_p^{t} = Z_p + \text{MSA}_{\text{temporal}}(\text{LN}(Z_p))
$$

where $Z_p = [z_{1,p}, z_{2,p}, \ldots, z_{T,p}]$ collects the same position across all frames.

### Video Sampling Strategies

| Strategy | Frames | Coverage | Best For |
|----------|--------|---------|----------|
| Uniform | $T$ frames equally spaced | Full video | Actions, scene classification |
| Dense | Every frame in a clip | Short segment | Fine-grained motion |
| Keyframe | Scene change detection | Key moments | Highlight detection |
| Multi-scale | $T_1$=4 (coarse) + $T_2$=16 (fine) | Both | Complex activities |

---

## Part 4: ImageBind — 6 Modalities, One Space

### The Core Idea

Instead of training pairwise models (image-text, image-audio, etc.), ImageBind uses **images as the binding modality**:

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     ImageBind Architecture                    │
  │                                                               │
  │      Depth ──► [Encoder] ──► ê_d ──┐                         │
  │                                     │                         │
  │      Audio ──► [Encoder] ──► ê_a ──┤                         │
  │                                     │  All aligned to          │
  │      Text  ──► [Encoder] ──► ê_t ──┤  IMAGE embedding         │
  │                                     │  space via contrastive   │
  │     Thermal ──►[Encoder] ──► ê_th──┤  learning                │
  │                                     │                         │
  │      IMU   ──► [Encoder] ──► ê_m ──┤                         │
  │                                     │                         │
  │     IMAGE  ──► [Encoder] ──► ê_I ──┘  (anchor modality)      │
  └──────────────────────────────────────────────────────────────┘
```

### Training Objective

Only **image-paired** contrastive losses:

$$
\mathcal{L}_{\text{ImageBind}} = \mathcal{L}_{\text{img-txt}} + \mathcal{L}_{\text{img-aud}} + \mathcal{L}_{\text{img-depth}} + \mathcal{L}_{\text{img-thermal}} + \mathcal{L}_{\text{img-IMU}}
$$

**Emergent property:** Even though audio-text pairs were never seen during training, audio and text become aligned **through** the shared image space (transitivity of alignment):

$$
\text{sim}(\text{audio}, \text{text}) \approx \text{sim}(\text{audio}, \text{image}) \times \text{sim}(\text{image}, \text{text})
$$

### Modality Encoders in ImageBind

| Modality | Encoder | Input | Embedding Dim |
|----------|---------|-------|--------------|
| Image | ViT-H/14 | 224×224×3 | 1024 |
| Text | CLIP Text Transformer | Token sequence | 1024 |
| Audio | ViT (on mel-spec) | 2s mel spectrogram | 1024 |
| Depth | ViT (single channel) | 224×224×1 | 1024 |
| Thermal | ViT (single channel) | 224×224×1 | 1024 |
| IMU | Transformer | 6-axis × 2s | 1024 |

---

## Part 5: Three-Modal Contrastive Loss

Extending InfoNCE to 3 modalities (image + text + audio):

$$
\mathcal{L}_{3\text{-modal}} = \mathcal{L}_{\text{img-txt}} + \mathcal{L}_{\text{img-aud}} + \mathcal{L}_{\text{txt-aud}}
$$

Each component is the standard symmetric InfoNCE:

$$
\mathcal{L}_{\text{img-txt}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{S_{ii}^{v,t}/\tau}}{\sum_j e^{S_{ij}^{v,t}/\tau}} + \log\frac{e^{S_{ii}^{v,t}/\tau}}{\sum_j e^{S_{ji}^{v,t}/\tau}}\right]
$$

```
  Three-modal embedding space:

  ┌────────────────────────────────────┐
  │       Shared ℝᵈ Space              │
  │                                    │
  │   🖼️ v_dog ●──● t_"dog barking"   │
  │              \                     │
  │               ● a_bark_sound       │
  │                                    │
  │   🖼️ v_piano ●──● t_"piano music" │
  │               \                    │
  │                ● a_piano_melody    │
  │                                    │
  │   All three modalities aligned!    │
  └────────────────────────────────────┘
```

---

## What You'll Build

- Audio processing pipeline (waveform → STFT → mel spectrogram)
- CLAP-style audio-text contrastive model
- Temporal attention for video frames
- Three-modal CLIP (image + text + audio)
- Cross-modal retrieval: find images using audio queries

---

## Prerequisites

- Module 02 (CLIP, contrastive learning)
- Module 03 (InfoNCE loss)
- Basic signal processing concepts (frequency, sampling)

---

## Next Step

**[03_efficient_deployment](../03_efficient_deployment/)** — Ship your model: ONNX, quantization, torch.compile
