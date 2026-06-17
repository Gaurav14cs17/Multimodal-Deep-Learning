# 04 — Diffusion Models for Multimodal Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/04_diffusion_models/04_diffusion_models.ipynb)

> **Time:** ~50 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

Diffusion models power Stable Diffusion, DALL-E 2, and text-to-image generation:

- DDPM forward and reverse process (full derivation)
- DDIM accelerated sampling
- Latent diffusion (Stable Diffusion)
- Train a tiny 1D diffusion demo

---

## DDPM Forward Process — Derivation

### Step 1: Markov Chain

Define forward noising over $T$ steps:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

where $\beta_t \in (0, 1)$ is the noise schedule.

### Step 2: Reparameterization

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$. Then:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

**Proof sketch:** Apply recursion — each step scales by $\sqrt{\alpha_t}$ and adds Gaussian noise. After $t$ steps, signal scales by $\sqrt{\bar{\alpha}_t}$ and total noise variance is $1 - \bar{\alpha}_t$.

### Step 3: Sample at Arbitrary $t$

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

### Numerical Example (1D, $x_0 = 2.0$)

Linear schedule: $\beta_1 = 0.1$, $\beta_2 = 0.2$, $\beta_3 = 0.3$

| $t$ | $\bar{\alpha}_t$ | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ |
|-----|-----------------|------------------------|--------------------------|
| 1 | 0.9 | 0.949 | 0.316 |
| 2 | 0.72 | 0.849 | 0.529 |
| 3 | 0.504 | 0.710 | 0.704 |

At $t=3$ with $\epsilon = 0.5$:

$$
x_3 = 0.710 \times 2.0 + 0.704 \times 0.5 = 1.420 + 0.352 = 1.772
$$

Signal retained: 71%; noise fraction: 35%.

---

## Mathematical Proofs

### Proof: DDPM Forward Process — Closed Form $q(\mathbf{x}_t \mid \mathbf{x}_0)$

**Step 1 — Single-step Markov transition:**

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

Reparameterize: $\mathbf{x}_t = \sqrt{1-\beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_{t-1}$, $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

**Step 2 — Define $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:**

Apply recursion from $\mathbf{x}_0$:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}
$$

**Why:** Product of Gaussians with scaling — variance adds: $(1-\alpha_1) + \alpha_1(1-\alpha_2) + \cdots = 1 - \bar{\alpha}_t$.

**Step 3 — Closed-form distribution:**

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

**∎**

#### Numerical Example

From table above: $x_0 = 2.0$, $t=3$, $\epsilon = 0.5$: $x_3 = 0.710 \times 2.0 + 0.704 \times 0.5 = 1.772$.

---

### Proof: ELBO for Diffusion Models — Variational Lower Bound

**Step 1 — Log-likelihood decomposition:**

$$
\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)}\left[\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)}\right] = \mathcal{L}_{\text{ELBO}}
$$

**Step 2 — Expand as sum over timesteps:**

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_q\left[-\text{KL}(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T)) + \sum_{t=2}^{T} \text{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)\right]
$$

**Step 3 — Simplify via noise prediction:** Each KL term reduces to $\mathbb{E}\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \rVert^2$ up to constants.

**Step 4 — Training objective (Ho et al.):**

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

**∎**

#### Numerical Example

1D: $x_0 = 1.0$, $t=100$, $\bar{\alpha}_t = 0.5$, $\epsilon_{\text{true}} = 0.3$, $\epsilon_\theta = 0.5$: loss $= (0.3 - 0.5)^2 = 0.04$.

---

### Proof: Score Matching Equivalence to Denoising

**Step 1 — Score function:** $\nabla_{\mathbf{x}} \log p(\mathbf{x}_t)$.

**Step 2 — For Gaussian perturbation:** $p(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$.

**Step 3 — Score of noisy distribution:**

$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t} = -\frac{\sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}}{1-\bar{\alpha}_t}
$$

**Step 4 — Equivalence:** Predicting noise $\boldsymbol{\epsilon}_\theta$ is equivalent to predicting the score — denoising IS score matching. **∎**

#### Numerical Example

$\mathbf{x}_t = 1.772$, $\bar{\alpha}_t = 0.504$, $x_0 = 2.0$, $\epsilon = 0.5$: score $\propto -(1.772 - 0.710 \times 2.0)/0.496 = -0.5/\sqrt{1-0.504}$ — matches injected noise direction.

---

## Reverse Process — Derivation

### Step 1: True Reverse (Intractable)

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\boldsymbol{\mu}_q, \tilde{\beta}_t \mathbf{I})
$$

where:

$$
\boldsymbol{\mu}_q = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
$$

### Step 2: Predict Noise

Train neural network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ to predict the noise added at step $t$:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

### Step 3: Sampling Step

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}
$$

**Numerical example:** $x_t = 1.772$, $t=3$, predicted $\hat{\epsilon} = 0.48$, $\alpha_3 = 0.7$, $\bar{\alpha}_3 = 0.504$:

$$
x_2 = \frac{1}{\sqrt{0.7}}\left(1.772 - \frac{0.3}{\sqrt{0.496}} \times 0.48\right) + \sigma_3 z \approx 1.89 + \text{noise}
$$

---

## DDIM — Deterministic Sampling

Skip stochastic term ($\sigma_t = 0$):

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}\right)}_{\hat{\mathbf{x}}_0} + \sqrt{1-\bar{\alpha}_{t-1}}\,\boldsymbol{\epsilon}_\theta
$$

Enables 50-step sampling instead of 1000 — critical for Stable Diffusion inference.

---

## Latent Diffusion (Stable Diffusion)

### Step 1: Encode to Latent

$$
\mathbf{z} = \mathcal{E}(\mathbf{x}), \quad \mathbf{x} \approx \mathcal{D}(\mathbf{z})
$$

VAE compresses $512 \times 512 \times 3$ → $64 \times 64 \times 4$ (8× spatial compression).

### Step 2: Diffuse in Latent Space

Apply DDPM to $\mathbf{z}$ instead of pixels — 64× less compute per step.

### Step 3: Text Conditioning

$$
\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) \quad \text{where } \mathbf{c} = \text{CLIP}(\text{text})
$$

**Classifier-Free Guidance:**

$$
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset) + w \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset))
$$

$w = 7.5$ typical — amplifies text influence.

---

## What You'll Build

- 1D Gaussian diffusion forward/reverse demo
- Noise prediction MLP trained on synthetic data
- Sampling trajectory visualization

---

## Next Step

**[05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb](../05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb)**
