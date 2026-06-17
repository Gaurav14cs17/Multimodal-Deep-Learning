# 04 — Batch Size, Memory Management & Mixed Precision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/04_batch_size_memory/04_batch_size_memory.ipynb)

> **Time:** ~55 minutes | **Difficulty:** Intermediate–Advanced | **GPU Required:** Optional (CPU demos work)

---

## What You'll Learn

- Small vs large batch effects on gradient variance and generalization
- Linear scaling rule derivation and when it breaks
- Gradient accumulation — mathematical equivalence to larger batch
- GPU memory breakdown: params, optimizer, activations, gradients
- Gradient checkpointing memory-compute tradeoff
- Mixed precision: FP32 vs FP16 vs BF16 (IEEE 754)
- Dynamic loss scaling algorithm
- AMP recipe: which ops stay FP32
- OOM solutions: accumulation, checkpointing, ZeRO, FSDP, CPU offload
- Code: memory estimator, batch variance demo, AMP loop

---

## Key Equations

### 1. Gradient Variance vs Batch Size

$$
\text{Var}[\nabla \mathcal{L}_B] = \frac{\sigma^2}{B}
$$

where $\sigma^2 = \text{Var}[\nabla \ell_i]$ for per-sample gradients.

**Numerical example:** $\sigma^2 = 1$, $B=1$ → Var=1; $B=256$ → Var=0.0039.

### 2. Linear Scaling Rule

$$
\eta_{new} = \eta_{base} \cdot \frac{B_{new}}{B_{base}}
$$

**Derivation:** Larger batch → lower gradient noise → can take larger steps. Goyal et al. (2017) with linear warmup.

**When it breaks:** Very large $B$ without LARS/LAMB; small dataset (few steps per epoch).

### 3. Gradient Accumulation

$$
g_{accum} = \frac{1}{K}\sum_{k=1}^{K} g_k
$$

Equivalent to batch size $B \times K$ if loss is averaged per micro-batch.

### 4. GPU Memory Budget

$$
\text{GPU\_Mem} = \underbrace{P \cdot b}_{params} + \underbrace{P \cdot b}_{grads} + \underbrace{k \cdot P \cdot b}_{optimizer} + \underbrace{B \cdot L \cdot d^2 \cdot b}_{activations}
$$

- $P$ = parameter count
- $b$ = bytes per element (4 for FP32, 2 for FP16)
- Adam: $k=2$ extra states ($m$, $v$) → total $3P$ for optimizer+params

### 5. Gradient Checkpointing

Trade memory $O(L)$ → $O(\sqrt{L})$ with ~33% extra forward passes (Chen et al.).

### 6. Dynamic Loss Scaling

```
if inf/nan in grad: scale /= 2
elif N steps without inf: scale *= 2
```

---

## ASCII — Memory Breakdown

```
GPU VRAM
┌─────────────────────────────────────┐
│ Model Parameters (P × 4 bytes)      │
├─────────────────────────────────────┤
│ Gradients (P × 4 bytes)             │
├─────────────────────────────────────┤
│ Optimizer states (Adam: 2P × 4)     │
├─────────────────────────────────────┤
│ Activations (B × L × d² × 4)  ← BIG │
└─────────────────────────────────────┘
         ↑ reduce B or checkpoint
```

---

## OOM Solution Decision Table

| Symptom | Solution | Tradeoff |
|---------|----------|----------|
| CUDA OOM at forward | Reduce batch size | Slower/noisier gradients |
| OOM, need large effective batch | Gradient accumulation | Same memory, more steps |
| OOM on activations | Gradient checkpointing | +33% compute |
| OOM on optimizer | ZeRO Stage 1/2 | Communication |
| OOM on everything | ZeRO-3 / FSDP | More communication |
| Still OOM | CPU offload (DeepSpeed) | Much slower |

---

## Paper References

- Goyal et al. (2017) — Large Minibatch SGD — [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- Micikevicius et al. (2018) — Mixed Precision — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Rajbhandari et al. (2020) — ZeRO — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

**Blog:** [Lilian Weng — Large Batch Training](https://lilianweng.github.io/posts/2021-12-05-large-batch/)
