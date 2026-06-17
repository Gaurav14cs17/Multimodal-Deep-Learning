# 05 — Training Instability, NaN/Inf & Convergence Issues

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/05_training_instability/05_training_instability.ipynb)

> **Time:** ~3 hours | **Difficulty:** Intermediate–Advanced | **GPU Required:** No (multi-GPU patterns run on CPU)

---

## What You'll Learn

- **IEEE 754 fundamentals:** why NaN exists (7 arithmetic rules), FP32/FP16/BF16 bit layouts
- **NaN propagation:** forward, backward, and AllReduce contagion — detect before spread
- **`torch.autograd.detect_anomaly()`:** the #1 debugging tool — traceback to exact op
- **15 common NaN sources:** log(0), KL div, cosine sim, LayerNorm, empty batch, focal loss, torch.where trap, GAN loss, and more — each with WHY/HOW/DEMO
- **Systematic 8-step debugging methodology** + `NaNTracer` utility (forward/backward hooks)
- **NaN prevention recipe table** by model type (CLIP, Transformer, GAN, LoRA, Diffusion, DDP, FSDP, ZeRO)
- Root cause analysis when loss becomes NaN
- Softmax overflow proof: $e^{88.7}$ exceeds FP32 max
- Log-sum-exp trick — full numerical equivalence proof
- Division by zero in normalization layers
- Attention logit explosion when $Q, K$ grow unbounded
- ASCII debugging flowchart for "My loss is NaN"
- Pre-LN vs Post-LN — gradient norm analysis
- μP (Maximal Update Parameterization) — width-independent hyperparameters
- Spectral normalization — Lipschitz constraint
- Gradient monitoring before NaN occurs
- Multimodal-specific: CLIP temperature collapse, modality collapse, catastrophic forgetting
- Code: NaN detector, stable softmax, CLIP temperature demo
- **Multi-GPU NaN:** FP16 overflow, loss scaling, gradient accumulation + AMP
- Cross-GPU NaN propagation via AllReduce, BatchNorm on small per-GPU batch
- NaN prevention toolkit: `NaNDetector`, `GradientSentinel`, `LossScaleMonitor`, `SafeTrainingLoop`
- 4 real-world debugging scenarios (CLIP, late NaN, rank-3 data poison, LLaVA+LoRA)

---

## IEEE 754 — Why NaN Exists

Every NaN comes from one of **7 IEEE 754 operations**:

| Operation | Result |
|-----------|--------|
| $0/0$ | NaN |
| $\infty - \infty$ | NaN |
| $\infty + (-\infty)$ | NaN |
| $0 \times \infty$ | NaN |
| $\infty / \infty$ | NaN |
| $\sqrt{\text{negative}}$ | NaN |
| Any op with existing NaN | NaN (propagation) |

**Bit layout:** NaN = all exponent bits = 1, mantissa $\neq 0$. Inf = all exponent bits = 1, mantissa = 0.

| dtype | bits | max |
|-------|------|-----|
| FP32 | 1+8+23 | $\approx 3.4 \times 10^{38}$ |
| FP16 | 1+5+10 | **65504** |
| BF16 | 1+8+7 | $\approx 3.4 \times 10^{38}$ (same range as FP32!) |

---

## 15 Common NaN Sources (Quick Reference)

| # | Source | Fix |
|---|--------|-----|
| 1 | `log(0)` in CE | `log_softmax`, clamp probs |
| 2 | KL divergence | clamp $q \geq \epsilon$, `F.kl_div` |
| 3 | Cosine sim, zero vector | `eps=1e-8` |
| 4 | LayerNorm zero variance | `eps >= 1e-5` (FP16: `1e-6`) |
| 5 | Out-of-range labels | validate `0 <= y < C` |
| 6 | Empty batch | skip if `batch.size(0) == 0` |
| 7 | OOV embedding index | `ids.clamp(0, vocab-1)` |
| 8 | Focal loss | `pred.clamp(1e-7, 1-1e-7)` |
| 9 | Large weight init | Kaiming/Xavier init |
| 10 | `torch.where` trap | clamp before division |
| 11 | `norm()` on zero vector | add epsilon inside norm |
| 12 | Attention FP16 overflow | BF16, flash attention |
| 13 | Residual grad explosion | Pre-LN, grad clip 1.0 |
| 14 | FP16 grad underflow | BF16, FP32 master weights |
| 15 | GAN D too strong | WGAN-GP, spectral norm |

---

## Systematic NaN Debugging (8 Steps)

```
1. REPRODUCE  → reduce data/model/GPUs
2. LOCATE     → torch.autograd.detect_anomaly()
3. TRACE      → NaNTracer forward/backward hooks
4. ISOLATE    → forward NaN vs backward NaN?
5. IDENTIFY   → match one of 15 sources above
6. FIX        → apply specific fix
7. VERIFY     → run 2× original NaN step count
8. HARDEN     → permanent NaN hooks in production
```

---

## Key Equations

### 1. Softmax Overflow (FP32)

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

When $x_i > 88.7$: $e^{x_i} > 1.6 \times 10^{38}$ → **overflow to Inf** → NaN after division.

### 2. Log-Sum-Exp Trick — Proof

Define $c = \max_i x_i$:

$$
\log\sum_i e^{x_i} = \log\left(e^c \sum_i e^{x_i - c}\right) = c + \log\sum_i e^{x_i - c}
$$

**Step 1:** Subtract $c$ → all exponents $\leq 0$ → no overflow.

**Step 2:** $\max_i (x_i - c) = 0$ → at least one term is $e^0 = 1$ → no underflow in sum.

**Numerical example:** $x = [1000, 1001, 999]$, $c=1001$ → stable log-sum = 1001 + log(1 + e^{-1} + e^{-2}).

### 3. Scaled Dot-Product Attention

$$
\text{Attn} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Scaling by $\sqrt{d_k}$ keeps logits $O(1)$ if $Q, K$ are unit variance. Without scaling or with growing weights → explosion.

### 4. Pre-LN vs Post-LN

**Post-LN block:** $x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))$

**Pre-LN block:** $x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))$ with LN **inside** sublayer input

Pre-LN: gradient norms more stable at init (Xiong et al. 2020).

### 5. Spectral Normalization

$$
\bar{W} = \frac{W}{\sigma(W)}, \quad \sigma(W) = \max_{\lVert h \rVert=1} \lVert Wh \rVert
$$

Ensures $\lVert \bar{W} h \rVert \leq \lVert h \rVert$ — Lipschitz constant ≤ 1.

### 6. CLIP Temperature Collapse

$$
\mathcal{L} = -\log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}
$$

As $\tau \to 0$: logits $s/\tau \to \pm\infty$ → softmax saturates → NaN in backward.

### 7. FP16 Overflow in AMP (Multi-GPU #1 Cause)

FP16 maximum $\approx 65504$. With dynamic loss scaling:

$$
\text{scaled\_loss} = s \cdot \mathcal{L}
$$

If activations or scaled loss exceed FP16 range → $\infty$ → NaN. On multi-GPU, one rank overflowing poisons AllReduce.

**BF16** uses FP32 exponent range — no 65504 cliff:

$$
\text{BF16 range} = \text{FP32 range}, \quad \text{mantissa bits} = 7 \text{ (vs 10 in FP16)}
$$

### 8. Dynamic Loss Scaling — Update Rules

On NaN/Inf gradient: $s \leftarrow s / 2$ (skip step).

After $N$ good steps: $s \leftarrow \min(2s,\; s_{\max})$.

When $s \to 1$ after repeated skips, effective update $\eta \cdot g / s$ vanishes → training stalls → eventual NaN.

### 9. Gradient Accumulation + AMP

Accumulate $K$ micro-batches with scaled loss. **Unscale once** after all $K$ backward passes:

$$
g_{\text{accum}} = \sum_{k=1}^{K} s \cdot \nabla \mathcal{L}_k \quad \Rightarrow \quad \text{unscale: } g = g_{\text{accum}} / s
$$

Unscaling inside the micro-step loop divides by $s$ multiple times → wrong gradient magnitude → NaN.

### 10. AllReduce NaN Propagation

$$
g = \frac{1}{G}\sum_{r=1}^{G} g_r \quad \text{— if } \exists r : g_r = \text{NaN} \Rightarrow g = \text{NaN on all ranks}
$$

**Gradient sentinel:** replace $\text{NaN}$ gradients with zero before sync.

### 11. Loss Spike → NaN on Multi-GPU

One rank with adversarial batch produces $g_{\text{bad}} \gg g_r$. After averaging:

$$
\left\lVert \frac{1}{G}\sum_r g_r \right\rVert \approx \left\lVert \frac{g_{\text{bad}}}{G} \right\rVert
$$

Still large unless per-rank clipping or batch skipping applied.

### 12. BatchNorm with Small Per-GPU Batch

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

When $B_{\text{local}} < 4$, one GPU may have $\sigma^2 \approx 0$ → division instability → NaN → SyncBatchNorm still fails if local stats are computed before sync incorrectly. Use **GroupNorm** or **LayerNorm** when $B_{\text{local}}$ is tiny.

### 13. FP16 Master Weight Desync

Smallest FP16 increment: $\Delta_{16} = 2^{-24} \approx 5.96 \times 10^{-8}$.

When $\lvert \eta \cdot g \rvert < \Delta_{16}$:

$$
W_{16} = \text{cast}(W_{32}) \text{ unchanged after FP32 update}
$$

Leads to weight stagnation in forward (FP16) while FP32 master drifts → eventual overflow.

---

## Multi-GPU NaN Debugging Flowchart

```
NaN on multi-GPU?
├── Which step? (early vs late)
│   ├── Step 0-10: Initialization problem
│   │   └── Check: init scale, LR warmup
│   ├── Step 10-1000: LR/optimizer issue
│   │   └── Check: LR schedule, loss scale
│   └── Step 1000+: Data/numerical issue
│       └── Check: batch quality, softmax overflow
├── Which rank first?
│   ├── All ranks simultaneously: global issue (LR, loss function)
│   └── Single rank first: data issue on that rank, BatchNorm
├── Using AMP?
│   ├── FP16: likely overflow → try BF16
│   └── BF16: rare overflow → check loss function
├── Loss scale dropping to 1?
│   └── Too many NaN skips → lower LR, check data
└── Grad norm before NaN?
    └── If spike: clip grads, check data quality
```

---

## NaN Debugging Flowchart (Single-GPU)

```
Loss is NaN?
    |
    ├─ YES → Check last finite step
    |         |
    |         ├─ Grad norm spiked? → Lower LR, clip grads
    |         ├─ Softmax in loss? → Log-sum-exp, check logits
    |         ├─ LayerNorm/div? → Add epsilon, check zero var
    |         ├─ Attention? → Pre-LN, scale by sqrt(d_k)
    |         └─ Multimodal? → Check tau, modality balance
    |
    └─ NO → Monitor grad norms proactively
```

---

## NaN Prevention Recipes by Model Type

| Model Type | Common NaN Source | Prevention Recipe |
|------------|-------------------|-------------------|
| CLIP / Contrastive | τ collapse, FP16 overflow | Clamp τ≥0.01, BF16, grad clip 1.0 |
| Transformer LM | Attention overflow, Post-LN | Pre-LN, flash attention, BF16 |
| Image Captioning | log(0) in CE | Label smoothing, min caption len=1 |
| VQA | Soft CE zero probs | Clamp probs, log_softmax |
| LoRA Finetuning | Large α/r scaling | α=r, grad clip 1.0 |
| Diffusion | Noise schedule edge | Clamp σ, stable loss |
| GAN | −log(D(G(z))) with D→0 | WGAN-GP, spectral norm |
| Multi-GPU DDP | AllReduce propagation | GradientSentinel, data validation |
| FSDP | Mixed precision casting | bf16/fp32/fp32 |
| DeepSpeed ZeRO | Sharded grad NaN | safe_get_full_grad, grad clipping |

---

## Solution Decision Table

| Symptom | Cause | Fix | When |
|---------|-------|-----|------|
| NaN after softmax | Overflow | Log-sum-exp | Any softmax loss |
| NaN from log(0) | Confident wrong CE | log_softmax + nll_loss | Classification |
| NaN in KL loss | q≈0 or p=0 | clamp q, F.kl_div | Distillation/VAE |
| NaN in cosine sim | Zero-norm vector | eps=1e-8 | Contrastive |
| NaN in LN (FP16) | eps too small | eps≥1e-5 | FP16 training |
| NaN from empty batch | mean([]) | skip batch | Filtered dataloaders |
| NaN in focal loss | log(0) | clamp pred | Object detection |
| NaN in torch.where | Both branches computed | clamp denominator | Custom ops |
| NaN in transformer | Post-LN instability | Switch to Pre-LN | Deep transformers |
| Loss spikes then NaN | LR too high | Warmup + lower LR | First 1k steps |
| CLIP loss NaN | τ → 0 | Clamp τ ≥ 0.01 | Contrastive training |
| One modality ignored | Modality collapse | Balance loss weights | Multimodal fusion |
| Finetune forgets pretrain | Catastrophic forgetting | Lower LR, EWC, LoRA | Domain adaptation |
| NaN on 4+ GPUs, step 500 | FP16 + τ collapse | Clamp τ, BF16, grad clip | CLIP/contrastive DDP |
| NaN after 10k steps multi-GPU | Loss scale → 1, weight growth | Weight decay, BF16, monitor scale | Long AMP runs |
| NaN on one rank first | Corrupted batch on that rank | DataValidator, sentinel before AllReduce | Sharded dataloaders |
| BatchNorm NaN per-GPU | $B_{\text{local}} < 4$ | SyncBatchNorm or GroupNorm | DDP with small batch |
| Grad accum + AMP NaN | Unscale inside micro-loop | Unscale once after $K$ steps | Large effective batch |
| NCCL hang (not NaN) | Rank desync / conditional branch | Same collectives all ranks; NCCL_DEBUG | Multi-GPU debugging |

---

## Paper References

- Micikevicius et al. (2018) — Mixed Precision Training — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Ott et al. (2019) — fairseq: FP16 Training — [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Goyal et al. (2017) — Large Minibatch SGD — [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- Xiong et al. (2020) — Pre-LN Transformers — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Yang et al. (2022) — μP — [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- Miyato et al. (2018) — Spectral Normalization — [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)

**Blogs & Docs:**
- [Lilian Weng — Training Large Neural Networks](https://lilianweng.github.io/posts/2021-09-25-train-compute/)
- [HuggingFace — Debugging Mixed Precision](https://huggingface.co/docs/transformers/perf_train_gpu_one#mixed-precision-training)
- [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA — Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
