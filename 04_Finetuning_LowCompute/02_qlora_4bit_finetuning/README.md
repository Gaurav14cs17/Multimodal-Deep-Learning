# 02 — QLoRA: 4-bit Quantized Finetuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/02_qlora_4bit_finetuning/02_qlora_4bit_finetuning.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Intermediate–Advanced | **GPU Required:** Recommended

---

## What You'll Learn

**QLoRA = Quantized LoRA.** Finetune 7B models that normally need 40GB VRAM on a single consumer GPU (~4 GB).

- Quantization basics: fp32 → fp16 → int8 → int4 — visualized
- NF4 (Normal Float 4-bit) — quantile quantization for neural network weights
- Double quantization — quantize the quantization constants
- Paged optimizers — offload optimizer states to CPU RAM
- End-to-end QLoRA finetuning with HuggingFace

---

## The Three Innovations of QLoRA

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     QLoRA = 3 Key Ideas                       │
  │                                                               │
  │  1. NF4 Quantization                                         │
  │     Weights: fp16 → 4-bit NF4  (4× compression)             │
  │     Optimal for normally-distributed neural net weights      │
  │                                                               │
  │  2. Double Quantization                                      │
  │     Quantize the quantization constants too                  │
  │     Saves additional 0.37 bits per parameter                 │
  │                                                               │
  │  3. Paged Optimizers                                         │
  │     Optimizer states (Adam m, v) → CPU when GPU OOMs         │
  │     Unified memory for CPU-GPU page transfers               │
  └──────────────────────────────────────────────────────────────┘
```

---

## Memory Comparison — Why QLoRA Matters

```
  7B Model Finetuning Memory:

  Full FT (fp32):  ████████████████████████████████████████  112 GB
                   (weights:28GB + gradients:28GB + optimizer:56GB)

  Full FT (fp16):  ████████████████████████                   56 GB
                   (weights:14GB + gradients:14GB + optimizer:28GB)

  LoRA (fp16):     ████████████████                           28 GB
                   (weights:14GB + LoRA grads:~0.1GB + optimizer:14GB)

  QLoRA (NF4):     ████                                       4 GB  ← Free Colab T4!
                   (weights:3.5GB + LoRA in fp16 + paged optimizer)
```

| Method | Model Memory | Optimizer Memory | Total | GPU Needed |
|--------|-------------|-----------------|-------|-----------|
| Full FT (fp32) | 28 GB | 84 GB (Adam: $3 \times$ model) | 112 GB | 2× A100 80GB |
| Full FT (fp16) | 14 GB | 42 GB | 56 GB | 1× A100 80GB |
| LoRA (fp16) | 14 GB | ~0.3 GB (LoRA only) | ~15 GB | 1× A100 40GB |
| QLoRA (NF4) | 3.5 GB | ~0.2 GB (paged) | ~4 GB | 1× RTX 4090 / T4 |

---

## Innovation 1: NF4 (Normal Float 4-bit) Quantization

### Standard Uniform Quantization (INT4)

Maps values uniformly between min and max:

$$
q = \text{round}\!\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^b - 1)\right)
$$

**Problem:** Neural network weights follow a **normal distribution** $\mathcal{N}(0, \sigma^2)$, so most values cluster near zero. Uniform quantization wastes levels on sparse tails.

### NF4: Quantile Quantization (Optimal for Normal Distributions)

Place quantization levels at **equal-probability quantiles** of the normal distribution:

$$
q_i = \Phi^{-1}\!\left(\frac{2i+1}{2 \cdot 2^b}\right), \quad i = 0, 1, \ldots, 2^b - 1
$$

where $\Phi^{-1}$ is the inverse standard normal CDF (probit function).

**For NF4 ($b = 4$, 16 levels):**

```
  Standard Normal Distribution with NF4 Quantization Levels:

  Probability density
       │
  0.4  │           ████
       │         ██    ██
  0.3  │        █        █
       │       █          █
  0.2  │      █            █
       │     █              █
  0.1  │   ██                ██
       │ ██                    ██
  0.0  │█▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔█
       └──┬─┬┬┬┬┬──┬──┬┬┬┬┬─┬──┘
         q₀q₁ ... q₇  q₈... q₁₅

  Many levels near 0         Few levels in tails
  (where most weights are)   (where few weights are)
```

**The 16 NF4 levels** (for a standard normal):

$$
\{-1.00, -0.69, -0.52, -0.39, -0.28, -0.18, -0.09, 0.00, 0.08, 0.17, 0.27, 0.38, 0.51, 0.68, 0.99, \text{special}\}
$$

### Quantization and Dequantization

**Quantize** (store in 4 bits):

$$
\text{Quantize}(x) = \arg\min_i \lvert x/s - q_i \rvert
$$

**Dequantize** (restore to fp16 for computation):

$$
\text{Dequant}(i) = s \cdot q_i
$$

where $s$ is the **absmax scale factor** per block of 64 weights:

$$
s = \max(\lvert x_1 \rvert, \ldots, \lvert x_{64} \rvert)
$$

---

## Innovation 2: Double Quantization

The scale factors $s$ are stored per block of 64 weights. With $N$ weights:

- Number of blocks: $N/64$
- Scale factors: $N/64$ values in fp32 (4 bytes each) = $N/16$ bytes
- Overhead: 0.5 bits per parameter

**Double quantization** quantizes the scale factors too:

$$
s_{\text{quantized}} = \text{INT8}(s / s_{\text{global}})
$$

where $s_{\text{global}}$ is computed over 256 consecutive scale factors.

**Savings:**

| Component | Without Double Quant | With Double Quant |
|-----------|--------------------|--------------------|
| Weight bits | 4 | 4 |
| Scale overhead | 0.5 bits/param | 0.127 bits/param |
| **Total** | **4.5 bits/param** | **4.127 bits/param** |
| Savings for 7B | — | 0.37 × 7B = **325 MB** |

---

## Innovation 3: Paged Optimizers

When GPU runs out of memory during a forward/backward pass (spikes from activations), optimizer states are automatically **paged to CPU**:

```
  Normal:                    Paged:
  ┌─────────────────┐       ┌─────────────────┐
  │     GPU VRAM     │       │     GPU VRAM     │
  │                  │       │                  │
  │ Weights (NF4)   │       │ Weights (NF4)   │
  │ LoRA A, B (fp16)│       │ LoRA A, B (fp16)│
  │ Gradients       │       │ Gradients       │
  │ Optimizer (m,v) │       │ Activations     │
  │ Activations     │       │                  │
  │                  │       └────────┬─────────┘
  │ OUT OF MEMORY!  │                │ Page transfer
  └─────────────────┘       ┌────────▼─────────┐
                            │     CPU RAM       │
                            │ Optimizer (m,v)  │
                            └─────────────────┘
```

Uses NVIDIA's **unified memory** (`cudaMallocManaged`) for automatic page transfers.

---

## QLoRA Forward Pass

During training, weights are dequantized on-the-fly:

$$
h = \underbrace{\text{Dequant}(W_{\text{NF4}})}_{\text{4-bit → fp16}} \cdot x + \frac{\alpha}{r} \underbrace{B A x}_{\text{LoRA in fp16}}
$$

```
  Forward pass:
  ┌──────────────┐     ┌──────────────────────┐
  │ W (NF4, 4bit)│ ──► │ Dequant to fp16      │ ──► W_fp16 · x
  └──────────────┘     │ (per-block, on-chip)  │         │
                       └──────────────────────┘         │
                                                         + (add)
  x ──► A (fp16) ──► Ax ──► B (fp16) ──► (α/r)BAx ────┘
                                                         │
                                                         ▼
                                                    h = output
```

**Gradient flow:** Gradients flow only through the LoRA path ($A, B$) — the NF4 base weights are **frozen** and never updated.

---

## Practical Hyperparameters

| Hyperparameter | Recommended | Why |
|---------------|-------------|-----|
| Base precision | NF4 | Optimal for normal weights |
| LoRA precision | fp16 (or bf16) | Gradients need higher precision |
| Rank $r$ | 16–64 | Higher than standard LoRA (compensates for frozen quantized base) |
| $\alpha$ | $2r$ | Standard scaling |
| Target modules | All linear | QLoRA benefits from more LoRA modules |
| Block size | 64 | Standard for NF4 |
| Double quant | True | Always enable (free 325MB) |
| Optimizer | Paged AdamW 8-bit | Further memory savings |

---

## Key Equations Summary

| Concept | Formula |
|---------|---------|
| NF4 levels | $q_i = \Phi^{-1}\!\left(\frac{2i+1}{2 \cdot 2^b}\right)$ |
| Absmax scale | $s = \max(\lvert x_1 \rvert, \ldots, \lvert x_{64} \rvert)$ |
| Dequantize | $\hat{x} = s \cdot q_{\text{nearest}}$ |
| QLoRA forward | $h = \text{Dequant}(W_{\text{NF4}}) \cdot x + \frac{\alpha}{r} BAx$ |
| Double quant savings | $0.37$ bits/param $\approx$ $325$ MB for 7B |
| Total bits/param | $4.127$ bits (with double quant) |

---

## Mathematical Proofs

### Proof: NF4 Optimality for Normally Distributed Weights

**Claim:** NF4 places quantization levels at equal-probability quantiles of $\mathcal{N}(0,1)$, minimizing expected quantization error for normally distributed weights.

**Step 1 — Weight distribution:** Neural network weights empirically follow $\mathcal{N}(0, \sigma^2)$ after training.

**Step 2 — Uniform INT4 problem:** Equal spacing wastes levels in low-density tails; too few levels near zero where most mass lies.

**Step 3 — Optimal quantile placement:**

$$
q_i = \Phi^{-1}\!\left(\frac{2i+1}{2 \cdot 2^b}\right), \quad i = 0, \ldots, 2^b - 1
$$

Each bin $[q_i, q_{i+1}]$ contains equal probability mass $1/2^b$ under $\mathcal{N}(0,1)$.

**Step 4 — Minimize $\mathbb{E}[(x - Q(x))^2]$:** For a given bit budget, equal-mass bins minimize MSE for any unimodal symmetric distribution (Lloyd-Max optimality for normal). **∎**

#### Numerical Example

8 weights $\mathcal{N}(0,1)$: $[-1.2, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5]$. NF4 assigns denser levels near 0; uniform INT4 step $= 0.5$ gives larger error on $[-0.1, 0.1]$ cluster.

---

### Proof: Quantization Error Bound for Uniform Quantization

**Claim:** For uniform quantizer with step size $\Delta$, $\lVert W - W_q \rVert_\infty \leq \Delta/2$.

**Step 1 — Quantization rule:**

$$
W_q[i] = \Delta \cdot \text{round}(W[i]/\Delta)
$$

**Step 2 — Per-element error:**

$$
\lvert W[i] - W_q[i] \rvert = \Delta \cdot \lvert W[i]/\Delta - \text{round}(W[i]/\Delta) \rvert \leq \Delta/2
$$

**Why:** Round maps to nearest grid point — maximum distance to grid is half a step.

**Step 3 — Block-wise absmax (QLoRA):** Scale $s = \max \lvert W \rvert$ over block of 64, $\Delta = 2s / 2^b$. Then $\lVert W - W_q \rVert_\infty \leq s / 2^b$ per block. **∎**

#### Numerical Example

Block max $s = 0.8$, 4-bit ($2^4 = 16$ levels): $\Delta = 1.6/16 = 0.1$. Weight $w = 0.37$ → quantized $0.4$, error $= 0.03 < 0.05 = \Delta/2$ ✓

---

## What You'll Build

- Visualize weight distributions (why normal assumption holds)
- Implement uniform vs NF4 quantization and compare error
- QLoRA finetuning with HuggingFace `BitsAndBytesConfig`
- Memory profiling: measure actual GPU usage at each stage
- Quality comparison: full FT vs LoRA vs QLoRA

---

## Prerequisites

- Notebook 01 (LoRA from scratch)
- Understanding of floating-point number representation
- Basic statistics (normal distribution, quantiles, CDF)

---

## 🔬 Worked Examples in the Notebook

### NF4 Quantization — Step-by-Step Numerical Trace
- Quantize 8 weight values from FP32 to NF4 (4-bit)
- Map each weight to nearest NF4 level (16 quantization points)
- Compute per-weight error, MSE, RMSE, and SNR
- Storage: 256 bits (FP32) → 32 bits (NF4) = 8x compression
- Verify: SNR > 20 dB confirms acceptable quality

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/02_qlora_4bit_finetuning/02_qlora_4bit_finetuning.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| QLoRA Memory Hierarchy | Dettmers et al. — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | 2023 | NF4 quantization + double quantization + paged optimizers |

### Key Innovation Detail

3 innovations that enable 65B model finetuning on single 48GB GPU:
1. **NF4 Quantization** — Information-theoretically optimal for normal distributions
2. **Double Quantization** — Quantize the quantization constants (saves 0.37 bits/param)
3. **Paged Optimizers** — CPU↔GPU memory swapping for gradient spikes

---

## Next Step

**[03_adapter_methods](../03_adapter_methods/)** — Beyond LoRA: Adapters, Prefix Tuning, IA3
