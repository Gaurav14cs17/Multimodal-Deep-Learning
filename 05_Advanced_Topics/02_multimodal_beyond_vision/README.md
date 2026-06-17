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

## Mathematical Proofs

### Proof: STFT — From DFT to Windowed Short-Time Analysis

**Step 1 — Discrete Fourier Transform (DFT) of frame $n$:**

$$
X(k) = \sum_{n=0}^{N-1} x(n) \, e^{-j2\pi kn/N}, \quad k = 0, \ldots, N-1
$$

**Why:** DFT decomposes a signal into frequency components.

**Step 2 — Localization via windowing:** Multiply signal by window $w(n)$ to analyze local segments:

$$
x_m(n) = x(n + mH) \cdot w(n)
$$

where $H$ is hop length, $m$ is frame index.

**Step 3 — STFT definition:**

$$
X(k, m) = \sum_{n=0}^{N-1} x(n + mH) \, w(n) \, e^{-j2\pi kn/N}
$$

**Why:** Windowing prevents spectral leakage; hopping gives time resolution.

**Step 4 — Power spectrogram:** $P(k,m) = \lvert X(k,m) \rvert^2$. **∎**

#### Numerical Example

16 kHz audio, $N=400$ (25 ms frame), $H=160$ (10 ms hop): 10 s audio $\approx$ 998 frames, 201 frequency bins ($N/2 + 1$).

---

### Proof: Mel Scale — Logarithmic Perceptual Mapping

**Step 1 — Psychoacoustic observation:** Pitch perception is approximately logarithmic in frequency.

**Step 2 — Mel definition:**

$$
m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)
$$

**Step 3 — Inverse mapping:**

$$
f = 700 \left(10^{m/2595} - 1\right)
$$

**Step 4 — Mel filterbank:** Triangular filters spaced uniformly on mel scale — more filters at low frequencies where human discrimination is finer. **∎**

#### Numerical Example

$f_1 = 100$ Hz → $m_1 = 2595 \log_{10}(1.143) \approx 100$ mel. $f_2 = 1000$ Hz → $m_2 \approx 999$ mel. Equal 100-mel spacing: $\Delta f \approx 43$ Hz at low freq vs $\Delta f \approx 430$ Hz at 1 kHz.

---

### Proof: Nyquist–Shannon Sampling Theorem (Sketch)

**Theorem:** A bandlimited signal with maximum frequency $f_{\max}$ can be perfectly reconstructed from samples at rate $f_s \geq 2 f_{\max}$.

**Step 1 — Sampling:** $x_s(n) = x(n/f_s)$.

**Step 2 — Spectrum replication:** Sampling convolves spectrum with impulses at multiples of $f_s$.

**Step 3 — No aliasing iff replicas don't overlap:** Requires $f_s/2 \geq f_{\max}$, i.e. $f_{\max} = f_s/2$ (Nyquist frequency).

**Step 4 — Speech at 16 kHz:** $f_{\max} = 8$ kHz — sufficient for telephony (300–3400 Hz) and most speech harmonics. **∎**

#### Numerical Example

Pure tone at 5 kHz sampled at 8 kHz ($f_s/2 = 4$ kHz): aliasing folds to 3 kHz — demonstrates why 16 kHz ($f_{\max}=8$ kHz) is needed for 4 kHz bandwidth.

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

## 🔬 Worked Examples in the Notebook

### Audio Processing — STFT from Scratch
- Generate synthetic audio signal with 3 frequencies (220, 440, 880 Hz)
- Compute STFT: 512-point FFT, Hann window, hop=160 samples
- Build Mel filterbank: 80 filters with denser coverage at low frequencies
- Convert to log-Mel spectrogram: [80, 101] — input to Whisper/CLAP
- 4-panel visualization: waveform, spectrogram, filterbank, mel spectrogram

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/02_multimodal_beyond_vision/02_multimodal_beyond_vision.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| ImageBind Overview (`../../assets/paper_figures/imagebind_overview.png`) | Girdhar et al. — [arXiv:2305.05665](https://arxiv.org/abs/2305.05665) | 2023 | 6-modality binding through shared embedding |
| Multimodal CoT (`../../assets/paper_figures/mm_cot.png`) | Zhang et al. — [arXiv:2302.00923](https://arxiv.org/abs/2302.00923) | 2023 | Two-stage rationale + answer inference |
| CLAP Architecture | Elizalde et al. — [arXiv:2206.04769](https://arxiv.org/abs/2206.04769) | 2023 | Audio-text contrastive pretraining + zero-shot classification |
| Audio SSL Models | Baevski / Hsu / Radford | 2020-23 | wav2vec 2.0, HuBERT, Whisper architectures |

### Advanced Topics — Key Papers & References

#### Multimodal Chain-of-Thought & Visual Reasoning

1. **Multimodal Chain-of-Thought Reasoning in Language Models** — Zhang et al. (2023) — [arXiv:2302.00923](https://arxiv.org/abs/2302.00923) — Two-stage rationale + answer inference
2. **VisProg: Visual Programming for Compositional Visual Reasoning** — Gupta & Kembhavi (2023) — [arXiv:2303.08128](https://arxiv.org/abs/2303.08128) — LLM-driven visual programming

#### Multimodal Hallucination

3. **Evaluating Object Hallucination in Large Vision-Language Models (POPE)** — Li et al. (2023) — [arXiv:2305.10355](https://arxiv.org/abs/2305.10355) — Systematic object hallucination benchmark
4. **HallusionBench: You See What You Think? Or You Think What You See?** — Guan et al. (2023) — [arXiv:2310.14566](https://arxiv.org/abs/2310.14566) — Image-context reasoning hallucination
5. **Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning (RLHF-V)** — Yu et al. (2023) — [arXiv:2306.14565](https://arxiv.org/abs/2306.14565) — RLHF for hallucination reduction

#### Evaluation Benchmarks

6. **MME: A Comprehensive Evaluation Benchmark for Multimodal LLMs** — Fu et al. (2023) — [arXiv:2306.13394](https://arxiv.org/abs/2306.13394)
7. **MMMU: A Massive Multi-Discipline Multimodal Understanding Benchmark** — Yue et al. (2023) — [arXiv:2311.16502](https://arxiv.org/abs/2311.16502)
8. **SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension** — Li et al. (2023) — [arXiv:2307.16125](https://arxiv.org/abs/2307.16125)
9. **MathVista: Evaluating Mathematical Reasoning in Visual Contexts** — Lu et al. (2023) — [arXiv:2310.02255](https://arxiv.org/abs/2310.02255)

#### Multimodal In-Context Learning

10. **Flamingo: a Visual Language Model for Few-Shot Learning** — Alayrac et al. (2022) — [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) — Interleaved image-text few-shot prompting
11. **MMICL: Empowering Vision-Language Model with Multi-Modal In-Context Learning** — Zhao et al. (2023) — [arXiv:2309.07915](https://arxiv.org/abs/2309.07915)
12. **Generative Pretraining in Multimodality (Emu)** — Sun et al. (2023) — [arXiv:2307.05222](https://arxiv.org/abs/2307.05222) — Unified multimodal pretraining

### Additional Papers Covered

- **wav2vec 2.0** (Baevski et al., 2020) — Contrastive learning on masked audio segments
- **HuBERT** (Hsu et al., 2021) — Offline clustering pseudo-labels + masked prediction
- **Whisper** (Radford et al., 2023) — Supervised ASR at scale (680K hours)
- **TimeSformer** (Bertasius et al., 2021) — Divided space-time attention for video
- **Video-LLaVA** (Lin et al., 2023) — Frame sampling + concatenation for video LLM
- **SAM** (Kirillov et al., 2023) — Foundation model for segmentation (SA-1B dataset)

---

## Next Step

**[03_efficient_deployment](../03_efficient_deployment/)** — Ship your model: ONNX, quantization, torch.compile
