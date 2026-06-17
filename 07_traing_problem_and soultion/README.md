# Module 07: Training Problems and Solutions

> **Time:** 4–6 hours | **Notebooks:** 6 | **Difficulty:** Intermediate–Advanced
>
> **This is a PRACTICAL TROUBLESHOOTING module** — when training breaks (NaN loss, OOM, divergence), this module tells you **why** mathematically and **how to fix it**.

| Notebook | Open in Colab |
|----------|---------------|
| `01_gradient_problems/01_gradient_problems.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/01_gradient_problems/01_gradient_problems.ipynb) |
| `02_loss_landscape_optimization/02_loss_landscape_optimization.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/02_loss_landscape_optimization/02_loss_landscape_optimization.ipynb) |
| `03_overfitting_underfitting/03_overfitting_underfitting.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/03_overfitting_underfitting/03_overfitting_underfitting.ipynb) |
| `04_batch_size_memory/04_batch_size_memory.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/04_batch_size_memory/04_batch_size_memory.ipynb) |
| `05_training_instability/05_training_instability.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/05_training_instability/05_training_instability.ipynb) |
| `06_distributed_training/06_distributed_training.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/06_distributed_training/06_distributed_training.ipynb) |

---

## Why This Module Matters

```
  You built the model (Modules 01–05)
              │
              ▼
  You chose the training strategy (Module 03)
              │
              ▼
  Training starts... then:
              │
    ┌─────────┼─────────┬─────────┬─────────┐
    ▼         ▼         ▼         ▼         ▼
  NaN loss   OOM     No conv.  Overfit   Slow
    │         │         │         │         │
    └─────────┴─────────┴─────────┴─────────┘
              │
              ▼
     Module 07 (THIS MODULE)
     Diagnose → Understand → Fix
```

Most multimodal projects fail not because the architecture is wrong, but because **gradients vanish**, **learning rates diverge**, **GPU memory runs out**, or **one modality collapses**. This module gives you the mathematical tools and code patterns to debug and fix each failure mode.

---

## Learning Path (Recommended Order)

```
01 Gradient Problems          ──► Foundation: why backprop fails deep
         │
         ▼
02 Loss Landscape & Optimizers ──► LR, Adam, schedulers
         │
         ▼
03 Overfitting & Regularization  ──► Generalization diagnostics
         │
         ▼
04 Batch Size & Memory           ──► Fit models on your hardware
         │
         ▼
05 Training Instability          ──► NaN/Inf, multimodal collapse
         │
         ▼
06 Distributed Training          ──► Scale beyond one GPU
```

**Prerequisites:** Modules 01 (foundations), 03 (training strategies). Helpful: Module 04 (finetuning).

---

## Quick Reference: Problem → Solution → When to Use

| Problem | Symptom | Solution | When to Use |
|---------|---------|----------|-------------|
| Vanishing gradients | Early layers don't learn; grad norms → 0 | Residual connections, proper init (He/Xavier), LayerNorm | Deep networks (>20 layers), transformers |
| Exploding gradients | Loss spikes; grad norms → ∞ | Gradient clipping, lower LR, spectral norm | RNNs, attention with large LR |
| LR too high | Loss oscillates or diverges | Reduce LR, warmup, cosine schedule | Any training; especially first 1k steps |
| LR too low | Loss plateaus early | Increase LR, cyclical LR, Adam | Stuck in sharp local minimum |
| Overfitting | Train ↓, val ↑ | Dropout, weight decay, augmentation, early stopping | Small dataset, large model |
| Underfitting | Train & val both high | More capacity, train longer, less regularization | Model too small for task |
| OOM (out of memory) | CUDA OOM error | Gradient accumulation, checkpointing, AMP, ZeRO | Large batch or model |
| Loss = NaN | Training crashes mid-run | `detect_anomaly()`, NaNTracer, log-sum-exp, Pre-LN | Any training — see notebook 05 Section 0 |
| **NaN from log(0), KL, focal loss** | NaN in loss backward | log_softmax, clamp probs, focal loss clamp | Classification, distillation, detection |
| **NaN from torch.where / norm** | NaN in backward only | Clamp denominator, eps in norm | Custom ops, contrastive |
| **NaN on multi-GPU (4+ GPUs)** | NaN on one rank poisons AllReduce | BF16, gradient sentinel, per-rank grad clip, data validation | DDP/FSDP + AMP |
| **NaN step 1–10 multi-GPU** | LR not scaled with GPU count | Linear LR scaling + warmup ∝ G | Any multi-GPU run |
| **NaN after 10k steps multi-GPU** | Loss scale → 1, FP16 overflow | Switch BF16, weight decay, monitor loss scale | Long AMP training |
| **NaN on single rank first** | Corrupted batch on one GPU | DataValidator, skip bad batches, sentinel before sync | Sharded dataloaders |
| **FSDP + checkpoint NaN** | Wrong dtype combo or wrapper order | MixedPrecision(bf16/fp32/fp32), FSDP outermost | Large model finetuning |
| CLIP temperature collapse | Similarities → ±∞ | Clamp τ, learnable τ with floor | Contrastive multimodal training |
| Modality collapse | One encoder dominates | Balanced loss weights, gradient surgery | Multimodal fusion models |
| Slow single-GPU training | Days per epoch | DDP, FSDP, mixed precision | Multi-GPU available |
| Large batch generalization gap | Train acc high, val low | Linear LR scaling + warmup, LARS/LAMB | Batch size > 256 |

---

## Sub-Topics

### [01 — Gradient Problems](01_gradient_problems/README.md)
Vanishing & exploding gradients: chain rule analysis, residual connections, normalization, initialization, gradient clipping.

### [02 — Loss Landscape & Optimization](02_loss_landscape_optimization/README.md)
Saddle points, learning rate schedules, SGD/Momentum, Adam/AdamW, LAMB/LARS, cosine annealing.

### [03 — Overfitting & Underfitting](03_overfitting_underfitting/README.md)
Bias-variance decomposition, dropout, weight decay, mixup, label smoothing, double descent.

### [04 — Batch Size & Memory](04_batch_size_memory/README.md)
Batch size effects, GPU memory breakdown, mixed precision, gradient accumulation, ZeRO, FSDP.

### [05 — Training Instability](05_training_instability/README.md)
**Definitive NaN resource:** IEEE 754 (7 creation rules, FP16/BF16 bit layouts), NaN propagation, `detect_anomaly()`, **15 common NaN sources** (log(0), KL, cosine sim, LayerNorm, empty batch, focal loss, torch.where, GAN, etc.), 8-step debugging methodology, `NaNTracer`, prevention recipes by model type, plus log-sum-exp, Pre-LN, μP, multi-GPU scenarios.

### [06 — Distributed Training](06_distributed_training/README.md)
DDP, FSDP, tensor/pipeline parallelism, AllReduce communication costs, scaling recipes, **LR linear scaling, warmup scaling, ZeRO/FSDP NaN debugging, distributed NaN prevention recipes (model-type × distributed table), launch checklist**.

---

## Key Papers (Module-Wide)

| Topic | Paper | Link |
|-------|-------|------|
| Initialization | Glorot & Bengio 2010 | [arXiv:1006.2785](https://arxiv.org/abs/1006.2785) |
| He Init | He et al. 2015 | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) |
| LayerNorm | Ba et al. 2016 | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) |
| Adam | Kingma & Ba 2015 | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| AdamW | Loshchilov & Hutter 2019 | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| Mixed Precision | Micikevicius et al. 2018 | [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) |
| fairseq FP16 | Ott et al. 2019 | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| Large Batch | Goyal et al. 2017 | [arXiv:1706.02677](https://arxiv.org/abs/1706.02677) |
| LARS | You et al. 2017 | [arXiv:1708.03888](https://arxiv.org/abs/1708.03888) |
| LAMB | You et al. 2020 | [arXiv:1904.00962](https://arxiv.org/abs/1904.00962) |
| Megatron-LM | Shoeybi et al. 2019 | [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) |
| ZeRO | Rajbhandari et al. 2020 | [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) |
| Pre-LN | Xiong et al. 2020 | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) |
| μP | Yang et al. 2022 | [arXiv:2203.03466](https://arxiv.org/abs/2203.03466) |

**Blog posts:** [Lilian Weng — Training Large Neural Networks](https://lilianweng.github.io/posts/2021-09-25-train-compute/) · [Lilian Weng — Large Batch Training](https://lilianweng.github.io/posts/2021-12-05-large-batch/) · [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html) · [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
