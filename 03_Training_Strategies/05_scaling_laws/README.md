# 05 — Scaling Laws for Multimodal Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/05_scaling_laws/05_scaling_laws.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

How to scale multimodal models compute-optimally:

- Chinchilla scaling laws (Hoffmann et al., 2022)
- Data vs model size trade-offs
- Multimodal scaling trends (CLIP, LLaVA)
- Fit and predict loss from compute budget

---

## Chinchilla Scaling Law — Derivation

### Step 1: Empirical Loss Model

Training loss as a function of model parameters $N$ and training tokens $D$:

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

where $E$ is irreducible loss, $A, B, \alpha, \beta$ are fitted constants.

Typical fitted values: $\alpha \approx 0.34$, $\beta \approx 0.28$, $E \approx 1.69$ (for language modeling).

### Step 2: Compute Budget

Training compute (FLOPs) approximately:

$$
C \approx 6 N D
$$

(6 FLOPs per parameter per token: forward + backward pass).

### Step 3: Optimal Allocation

Minimize $L(N, D)$ subject to $C = 6ND$:

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

**Key insight:** Scale **model size and data equally** — most labs previously under-trained large models.

### Numerical Example

Given compute budget $C = 10^{21}$ FLOPs:

$$
N_{\text{opt}} \approx 10^{9.5} \approx 3 \times 10^9 \text{ params}, \quad D_{\text{opt}} \approx \frac{C}{6N} \approx 5 \times 10^{10} \text{ tokens}
$$

Compare to GPT-3 (175B params, 300B tokens) — under-trained by Chinchilla standards (should use ~1.4T tokens).

---

## Loss Prediction — Worked Example

Using simplified constants $E=0.1$, $A=100$, $B=100$, $\alpha=\beta=0.5$:

$$
L(N, D) = 0.1 + \frac{100}{\sqrt{N}} + \frac{100}{\sqrt{D}}
$$

| $N$ (params) | $D$ (tokens) | $L(N,D)$ |
|-------------|-------------|----------|
| $10^6$ | $10^6$ | $0.1 + 0.1 + 0.1 = 0.30$ |
| $10^8$ | $10^8$ | $0.1 + 0.01 + 0.01 = 0.12$ |
| $10^9$ | $10^{10}$ | $0.1 + 0.003 + 0.001 = 0.104$ |

Doubling both $N$ and $D$ reduces loss faster than doubling only one.

---

## Multimodal Scaling — CLIP Trends

CLIP scaling (Radford et al., 2021):

| Model | Params | Image-Text Pairs | ImageNet Zero-Shot |
|-------|--------|-----------------|-------------------|
| RN50 | 102M | 400M | 59.8% |
| ViT-B/32 | 151M | 400M | 63.2% |
| ViT-L/14 | 428M | 400M | 75.5% |

**Observation:** Larger encoders + same data → better zero-shot. More data also helps (ALIGN: 1.8B noisy pairs).

Multimodal effective tokens:

$$
D_{\text{eff}} = D_{\text{text}} + \lambda \cdot D_{\text{image}}
$$

where $\lambda$ weights image-token equivalents (one image $\approx$ hundreds of text tokens in compute).

---

## Data Scaling vs Model Scaling

Fix compute $C = 6ND$. If we increase $N$ 4× without increasing $D$:

$$
D_{\text{new}} = \frac{C}{6 \cdot 4N} = \frac{D}{4}
$$

$$
\Delta L \approx \underbrace{-\frac{A}{4^\alpha N^\alpha}}_{\text{model gain}} + \underbrace{+\frac{B}{4^\beta D^\beta}}_{\text{data loss}} 
$$

For $\alpha \approx \beta$, gains cancel — **must scale both**.

**Numerical example:** $N=10^8$, $D=10^9$, $C=6\times10^{17}$.

4× model ($N=4\times10^8$) with fixed $C$: $D=2.5\times10^8$ (4× less data).

- Model term: $100/\sqrt{4\times10^8} = 0.005$ vs $100/\sqrt{10^8} = 0.01$ (gain 0.005)
- Data term: $100/\sqrt{2.5\times10^8} = 0.0063$ vs $100/\sqrt{10^9} = 0.0032$ (loss 0.0031)

Net: worse overall loss despite bigger model.

---

## Mathematical Proofs

### Proof: Chinchilla Optimal Allocation — Full Derivation

**Step 1 — Loss model:**

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

**Step 2 — Compute budget constraint:**

$$
C = 6ND \implies D = \frac{C}{6N}
$$

**Step 3 — Substitute into loss:**

$$
L(N) = E + \frac{A}{N^\alpha} + \frac{B}{(C/6N)^\beta} = E + AN^{-\alpha} + B' N^{\beta}
$$

where $B' = B(6/C)^\beta$.

**Step 4 — Minimize w.r.t. $N$:**

$$
\frac{dL}{dN} = -\alpha A N^{-\alpha-1} + \beta B' N^{\beta-1} = 0
$$

$$
N^{\alpha+\beta} = \frac{\alpha A}{\beta B'} \implies N_{\text{opt}} \propto C^{\frac{1}{\alpha+\beta}}
$$

With $\alpha \approx 0.34$, $\beta \approx 0.28$: $N_{\text{opt}} \propto C^{0.5}$ and $D_{\text{opt}} = C/(6N) \propto C^{0.5}$.

**Step 5 — Key insight:** Scale model and data equally — under-training large models wastes compute. **∎**

#### Numerical Example

$C = 10^{21}$ FLOPs: $N_{\text{opt}} \approx 3 \times 10^9$ params, $D_{\text{opt}} \approx 5 \times 10^{10}$ tokens. GPT-3 (175B, 300B tokens) is over-parameterized for its data budget.

---

## What You'll Build

- Chinchilla loss predictor $L(N, D)$
- Compute-optimal $(N, D)$ calculator for budget $C$
- Plot iso-loss curves and scaling trends

---

## Next Step

**[04_Finetuning_LowCompute/01_lora_from_scratch/01_lora_from_scratch.ipynb](../../04_Finetuning_LowCompute/01_lora_from_scratch/01_lora_from_scratch.ipynb)**
