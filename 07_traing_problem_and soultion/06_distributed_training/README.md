# 06 — Distributed Training — Data, Model & Pipeline Parallel

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/06_distributed_training/06_distributed_training.ipynb)

> **Time:** ~3 hours | **Difficulty:** Advanced | **GPU Required:** No (CPU demos; multi-GPU optional)

---

## What You'll Learn

- **Distributed NaN prevention recipes:** DDP AllReduce poisoning, FSDP dtype combos, ZeRO sharded NaN
- **Comprehensive model-type × distributed table:** CLIP, Transformer, LoRA+FSDP, pipeline parallel
- **Launch checklist:** LR scaling, warmup scaling, NCCL debug env, BF16, grad clip, sentinel
- Why distributed training: time scaling formula
- Data parallelism — AllReduce gradient averaging
- Ring AllReduce bandwidth derivation
- DDP — gradient bucketing, overlap with backward
- FSDP — fully sharded data parallel
- Tensor parallelism — split weight matrices
- Pipeline parallelism — GPipe, bubble ratio formula
- Sequence parallelism for long contexts
- Communication cost formulas
- Practical recipes: 1 GPU → 64+ GPUs
- Code: AllReduce comm estimator, DDP skeleton, scaling analysis
- **Multi-GPU NaN debugging:** LR linear scaling, warmup scaling with GPU count
- DeepSpeed ZeRO NaN inspection, FSDP mixed precision safe/unsafe combos
- Gradient checkpointing + FSDP + AMP wrapper order
- LARS/LAMB when batch > 8k, distributed debugging scenarios

---

## Key Equations

### 1. Training Time Scaling

$$
T \propto \frac{N_{steps} \cdot B_{local}}{G}
$$

$G$ = GPU count, $B_{local}$ = per-GPU batch size. Ideal: linear speedup in $G$.

### 2. Data Parallel Gradient Sync

$$
g = \frac{1}{G}\sum_{g=1}^{G} g_g
$$

Each GPU computes $g_g$ on local batch; AllReduce produces global average.

### 3. Ring AllReduce Communication

$$
\text{Bytes transferred} = \frac{2(G-1)}{G} \cdot P \cdot \text{bytes\_per\_param}
$$

**Derivation sketch:** Ring passes $P/G$ bytes per step; $2(G-1)$ steps to reduce and broadcast.

**Numerical example:** $P=7B$, FP32, $G=8$ → $\frac{14}{8} \times 7 \times 10^9 \times 4 \approx 49$ GB per step.

### 4. Pipeline Bubble Ratio

$$
\text{bubble fraction} = \frac{G-1}{G-1+M}
$$

$G$ = pipeline stages, $M$ = micro-batches. More micro-batches → smaller bubble.

**Example:** $G=4$, $M=8$ → bubble = 3/11 ≈ 27%.

### 5. Tensor Parallelism

Split weight matrix: $W = [W_1 \mid W_2]$ across 2 GPUs:

$$
Y = XW = [XW_1 \mid XW_2]
$$

Requires AllReduce or AllGather on partial outputs.

### 6. Learning Rate Linear Scaling (Critical for Avoiding NaN)

$$
\eta_{\text{new}} = \eta_{\text{base}} \times \frac{B_{\text{global}}}{B_{\text{base}}}
$$

**Example:** 4 GPUs × batch 32 = global batch 128, base batch 32 → multiply LR by **4×**.

Without scaling: larger effective batch + same LR → loss spikes → NaN at steps 10–1000.

### 7. Warmup Scaling with GPU Count

$$
\text{warmup\_steps} \propto G \quad \text{or} \quad \text{warmup\_steps} \propto \sqrt{G}
$$

More GPUs → larger global batch → need longer warmup to avoid immediate divergence (NaN at steps 1–10).

### 8. FSDP Mixed Precision — Safe Combinations

$$
\text{Forward: } W_{\text{bf16}} x \rightarrow \text{reduce in FP32}
$$

| param_dtype | reduce_dtype | buffer_dtype | Verdict |
|-------------|--------------|--------------|---------|
| bfloat16 | float32 | float32 | **Safe** |
| float16 | float32 | float32 | OK with GradScaler |
| float16 | float16 | float16 | **NaN prone** |

### 9. ZeRO Gradient Partitioning & NaN

With ZeRO Stage 2/3, gradients are sharded — NaN may appear on one rank's shard only. Use `deepspeed.utils.safe_get_full_grad` to inspect full gradient before optimizer step.

---

## Multi-GPU NaN Debugging (Distributed)

### Symptom → Cause → Fix

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| NaN at step 1–10 on multi-GPU | LR not scaled + short warmup | Linear LR scale; warmup $\propto G$ |
| NaN at step 500, CLIP 4-GPU | FP16 overflow + τ collapse | BF16, clamp τ, grad clip |
| NaN after 10k steps, loss scale=1 | FP16 weight/activation growth | BF16, weight decay, monitor scale |
| NaN on rank 3 first | Corrupted batch on one shard | DataValidator, sentinel before AllReduce |
| NaN with FSDP + checkpoint | Wrong wrapper order / dtypes | FSDP outermost; `use_reentrant=False` |
| NaN with ZeRO Stage 3 | Hidden NaN in partition | `safe_get_full_grad`, grad clipping |
| Hang (not NaN) | Rank desync, NCCL timeout | Same collectives all ranks; NCCL_DEBUG=INFO |

### Debugging Checklist

```
Distributed NaN?
├── Scaled LR with GPU count?
├── Warmup scaled with G?
├── FSDP MixedPrecision safe combo?
├── Grad checkpoint use_reentrant=False?
├── ZeRO: safe_get_full_grad check?
├── All ranks same code path?
├── NCCL_DEBUG=INFO for hangs
└── Per-rank grad norm before AllReduce
```

### Environment Variables

```bash
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## ASCII — Parallelism Strategies

```
DATA PARALLEL (DDP):
  GPU0: model copy, batch B0 ──┐
  GPU1: model copy, batch B1 ──┼── AllReduce(grads) ──> sync update
  GPU2: model copy, batch B2 ──┘

TENSOR PARALLEL:
  GPU0: W[:, :d/2]  ──┐
  GPU1: W[:, d/2:]  ──┴── AllGather output

PIPELINE PARALLEL:
  GPU0: layers 1-8  ──> GPU1: layers 9-16 ──> GPU2: layers 17-24
         micro-batch pipeline (reduces idle time)
```

---

## Scaling Recipe Table

| Hardware | Strategy | NaN Prevention |
|----------|----------|----------------|
| 1 GPU | AMP + checkpointing | BF16, grad clip |
| 2–8 GPUs | DDP + linear LR scale | Warmup ∝ G, gradient sentinel |
| Large model, 2–8 GPUs | FSDP | MixedPrecision(bf16/fp32/fp32) |
| 8–64 GPUs | FSDP + tensor parallel | LARS/LAMB if batch > 8k |
| 64+ GPUs | 3D parallel + DeepSpeed | ZeRO grad inspect, stable CE |

---

## Distributed NaN Prevention Recipes

| Scenario | NaN Mechanism | Distributed Fix |
|----------|---------------|-----------------|
| DDP AllReduce | One rank NaN → all ranks NaN | GradientSentinel; per-rank batch validation |
| FSDP sharding | NaN hidden in param shard | MixedPrecision(bf16, fp32, fp32) |
| ZeRO Stage 3 | Partial NaN in grad partition | `safe_get_full_grad`, grad clipping |
| LR not scaled | Effective batch ↑, LR same | $\eta_{new} = \eta_{base} \times B_{global}/B_{base}$ |
| Short warmup | NaN at step 1–10 | warmup_steps $\propto G$ |
| Grad accum + AMP | Unscale inside micro-loop | Unscale once after $K$ steps |
| BatchNorm DDP | $B_{local} < 4$ | SyncBatchNorm or GroupNorm |
| Pipeline parallel | NaN hard to locate | Log micro-batch ID + stage rank |
| Tensor parallel | Vocab-parallel CE overflow | Stable log-sum-exp CE |

### Model Type × Distributed (Full Recipe)

| Model Type | Distributed Amplifier | Full Prevention |
|------------|----------------------|-----------------|
| CLIP / Contrastive | FP16 overflow × AllReduce | Clamp τ, BF16, grad clip, LR scale × G |
| Transformer LM | Long seq × tensor parallel | Pre-LN, flash attention, BF16 |
| LoRA + FSDP | FSDP dtype mismatch | α=r, MixedPrecision(bf16/fp32/fp32) |
| GAN | D stronger on one GPU | WGAN-GP, spectral norm, sync updates |
| DeepSpeed ZeRO | Hidden NaN in partition | safe_get_full_grad, gradient_clipping=1.0 |

See notebook **05** for IEEE 754 fundamentals, 15 NaN sources, `detect_anomaly()`, and `NaNTracer`.

---

## Paper References

- Micikevicius et al. (2018) — Mixed Precision — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Ott et al. (2019) — fairseq FP16 — [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Goyal et al. (2017) — Large Minibatch SGD — [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- You et al. (2017) — LARS — [arXiv:1708.03888](https://arxiv.org/abs/1708.03888)
- You et al. (2020) — LAMB — [arXiv:1904.00962](https://arxiv.org/abs/1904.00962)
- Rajbhandari et al. (2020) — ZeRO — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Shoeybi et al. (2019) — Megatron-LM — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Li et al. (2020) — PyTorch DDP — [Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Narayanan et al. (2021) — Megatron-LM — [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- Zhao et al. (2023) — PyTorch FSDP — [Blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

**Blogs & Docs:**
- [Lilian Weng — Training Large Neural Networks](https://lilianweng.github.io/posts/2021-09-25-train-compute/)
- [HuggingFace — Debugging Mixed Precision](https://huggingface.co/docs/transformers/perf_train_gpu_one#mixed-precision-training)
- [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA — Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
