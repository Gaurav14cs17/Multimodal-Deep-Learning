# 07 — Text-to-Image & Video Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/07_text_to_image_video/07_text_to_image_video.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

How text becomes images and video:

- DALL-E autoregressive vs Stable Diffusion latent approach
- Classifier-free guidance (CFG) — full derivation
- Video generation via temporal attention
- Architecture comparison

---

## Classifier-Free Guidance — Full Derivation

### Step 1: Conditional Score

Diffusion model learns $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})$ where $\mathbf{c}$ is text embedding.

### Step 2: Unconditional Score

Also train with $\mathbf{c} = \emptyset$ (empty prompt) with probability $p_{\text{drop}}$ (typically 0.1–0.2).

### Step 3: Guided Score

Ho & Salimans (2022) interpolate:

$$
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset) + w \cdot \left(\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset)\right)
$$

where $w \geq 1$ is guidance weight.

**Interpretation:** Push noise prediction toward conditional direction, away from unconditional.

### Numerical Example (1D latent)

$\boldsymbol{\epsilon}_\text{uncond} = 0.3$, $\boldsymbol{\epsilon}_\text{cond} = 0.8$, $w = 7.5$:

$$
\tilde{\epsilon} = 0.3 + 7.5 \times (0.8 - 0.3) = 0.3 + 3.75 = 4.05
$$

Strong guidance amplifies text influence 7.5× beyond the conditional-unconditional gap.

| $w$ | Effect |
|-----|--------|
| 1.0 | No guidance (standard conditional) |
| 7.5 | Stable Diffusion default — sharp, text-faithful |
| 15+ | Oversaturated, artifact-prone |

---

## DALL-E vs Stable Diffusion

| Aspect | DALL-E 2/3 | Stable Diffusion |
|--------|-----------|------------------|
| Representation | Discrete tokens (VQ-VAE) or diffusion | Continuous latent (VAE) |
| Generation | Autoregressive or diffusion | Latent diffusion |
| Resolution | Up to 1024² | 512² native, upscaled |
| Text encoder | CLIP / custom | CLIP / OpenCLIP |
| Compute | High (full pixel space) | Lower (64×64 latent) |

---

## VQ-VAE Tokenization (DALL-E) — Derivation

### Step 1: Encode to Discrete Codes

$$
\mathbf{z} = \text{Quantize}(\mathcal{E}(\mathbf{x})) \in \{1, \ldots, K\}^{h \times w}
$$

### Step 2: Autoregressive Prior

$$
P(\mathbf{z} \mid \mathbf{c}) = \prod_{i=1}^{hw} P(z_i \mid z_{1:i-1}, \mathbf{c})
$$

Transformer predicts next spatial token given text $\mathbf{c}$.

**Numerical example:** $h \times w = 32 \times 32 = 1024$ tokens, vocab $K = 8192$:

- Sequence length: 1024 autoregressive steps
- Per-step: softmax over 8192 codes

---

## Video Generation — Temporal Attention

Extend 2D UNet with temporal dimension:

$$
\text{Attn3D}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

where $Q, K, V$ span spatial **and** temporal dimensions $(H \times W \times T)$.

**Factorized variant (VideoLDM):** Separate spatial and temporal attention:

$$
\text{Block} = \text{SpatialAttn}(\mathbf{x}) + \text{TemporalAttn}(\mathbf{x})
$$

Complexity: $O(T \cdot H^2 W^2 + H W \cdot T^2)$ vs $O(T^2 H^2 W^2)$ for full 3D attention.

**Numerical example:** $T=16$ frames, $H=W=32$:

- Full 3D: $(16 \times 32^2)^2 \approx 2.7 \times 10^8$ attention pairs
- Factorized: $16 \times 32^4 + 32^2 \times 16^2 \approx 1.7 \times 10^7$ (16× cheaper)

---

## CLIP Guidance (Pre-CFG era)

Original approach — gradient from CLIP similarity:

$$
\mathbf{z}_{t-1} = \mathbf{z}_t - \eta \nabla_{\mathbf{z}_t} \mathcal{L}_{\text{CLIP}}(\mathbf{z}_t, \text{text})
$$

where $\mathcal{L}_{\text{CLIP}} = -\cos(\text{CLIP}_{\text{img}}(\mathbf{z}), \text{CLIP}_{\text{text}}(\text{prompt}))$.

CFG replaced this with simpler dual forward pass — no gradient through CLIP needed.

---

## Mathematical Proofs

### Proof: Classifier-Free Guidance — Score Derivation

**Step 1 — Conditional score (Song et al.):** $\nabla_{\mathbf{z}} \log p(\mathbf{z}_t \mid \mathbf{c})$.

**Step 2 — Bayes' rule for conditional generation:**

$$
\nabla \log p(\mathbf{z}_t \mid \mathbf{c}) = \nabla \log p(\mathbf{z}_t) + \nabla \log p(\mathbf{c} \mid \mathbf{z}_t)
$$

**Step 3 — Approximate with noise predictors:** $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})$ is conditional; $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset)$ is unconditional.

**Step 4 — Guided prediction (Ho & Salimans):

$$
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset) + w \cdot \left(\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset)\right)
$$

**Why:** Amplifies the conditional direction $(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ by factor $w$ — stronger text adherence. **∎**

#### Numerical Example

$\epsilon_{\text{uncond}} = 0.3$, $\epsilon_{\text{cond}} = 0.8$, $w = 7.5$: $\tilde{\epsilon} = 0.3 + 7.5 \times 0.5 = 4.05$.

---

### Proof: Latent Diffusion — Why Diffuse in VAE Space

**Step 1 — Pixel diffusion cost:** $512 \times 512 \times 3$ dimensions per step — prohibitive.

**Step 2 — VAE encode:** $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{64 \times 64 \times 4}$ — 8× spatial compression, 48× fewer dimensions per step.

**Step 3 — Diffusion in latent space:**

$$
q(\mathbf{z}_t \mid \mathbf{z}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{z}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

**Step 4 — Decode:** $\mathbf{x} = \mathcal{D}(\mathbf{z}_0)$ after reverse sampling. **∎**

#### Numerical Example

Pixel space: $512^2 \times 3 \approx 786$K dims. Latent: $64^2 \times 4 = 16$K dims — **49× reduction** per diffusion step.

---

## What You'll Build

- CFG sampling demo on synthetic 8-dim latent vectors
- Architecture comparison diagram
- Guidance weight sweep visualization

---

## Next Step

**[08_multimodal_agents/08_multimodal_agents.ipynb](../08_multimodal_agents/08_multimodal_agents.ipynb)**
