# 03 — Efficient Deployment: Ship Your Multimodal Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/03_efficient_deployment/03_efficient_deployment.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

You've built and trained the model — now **ship it**:

- ONNX export — run anywhere (CPU, GPU, edge, mobile)
- Post-training quantization (INT8) — 4x smaller, 2x faster
- Knowledge distillation — compress 7B → 1B
- `torch.compile` — 1-line 2x speedup
- Benchmarking: measure latency, throughput, memory

---

## Deployment Decision Tree

```
  Your trained model
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  What's your deployment target?                      │
  │                                                      │
  │  ├── Mobile / Edge / Browser?                        │
  │  │   └──► ONNX Export + INT8 Quantization           │
  │  │        (smallest, most portable)                  │
  │  │                                                   │
  │  ├── GPU Server (high throughput)?                   │
  │  │   └──► TensorRT or torch.compile                 │
  │  │        (fastest GPU inference)                    │
  │  │                                                   │
  │  ├── GPU Server (LLM / VLM)?                        │
  │  │   └──► vLLM + Continuous Batching                │
  │  │        (optimal for autoregressive generation)    │
  │  │                                                   │
  │  ├── CPU Server (cost-sensitive)?                    │
  │  │   └──► ONNX Runtime + INT8                       │
  │  │        (no GPU needed, good latency)              │
  │  │                                                   │
  │  └── Quick prototype?                                │
  │      └──► torch.compile (1 line, 2× speedup)        │
  └─────────────────────────────────────────────────────┘
```

---

## Technique 1: ONNX Export

### What is ONNX?

ONNX (Open Neural Network Exchange) is an **intermediate representation** that lets you run models on any hardware with any runtime:

```
  PyTorch Model
       │
       ▼  torch.onnx.export()
  ┌──────────────┐
  │  ONNX Graph   │  (hardware-agnostic IR)
  │  .onnx file   │  Contains: nodes, edges, shapes, dtypes
  └──────┬───────┘
         │
    ┌────┴─────────────────────┐
    │         │                │
    ▼         ▼                ▼
  ONNX      TensorRT        CoreML
  Runtime   (NVIDIA GPU)     (Apple)
  (CPU/GPU)
```

### Export Process

The key equation for tracing:

$$
\text{ONNX}(f) = \text{Trace}(f, x_{\text{dummy}}) \to \text{Static Graph}
$$

**Important:** ONNX tracing requires **static shapes** — dynamic control flow (`if`, variable-length loops) needs special handling with `torch.onnx.export(..., dynamic_axes={...})`.

### ONNX Graph Optimization Passes

| Pass | What It Does | Speedup |
|------|-------------|---------|
| Constant folding | Pre-compute static operations | 5-10% |
| Operator fusion | Merge Conv+BN+ReLU into single op | 10-30% |
| Dead code elimination | Remove unused nodes | Minor |
| Shape inference | Pre-compute tensor shapes | Minor |

---

## Technique 2: Post-Training Quantization (INT8)

### The Quantization Equation

Map floating-point values to 8-bit integers:

$$
q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z, \; 0, \; 255\right)
$$

**Dequantize** (approximate recovery):

$$
\hat{x} = s \cdot (q - z)
$$

where:
- $s$ = scale factor
- $z$ = zero point (integer offset)

### Scale Factor Computation

**Symmetric quantization** (commonly used for weights):

$$
s = \frac{\max(\lvert x \rvert)}{127}, \quad z = 0
$$

**Asymmetric quantization** (commonly used for activations):

$$
s = \frac{x_{\max} - x_{\min}}{255}, \quad z = \text{round}\!\left(-\frac{x_{\min}}{s}\right)
$$

### Calibration — How to Choose Scale Factors for Activations

Since activation ranges are data-dependent, we run a **calibration dataset** through the model:

```
  Calibration Process:
  ┌──────────────┐     ┌──────────────────────┐
  │  Calibration  │ ──► │  Forward pass         │
  │  Dataset      │     │  (100-500 samples)    │
  │  (no labels)  │     │  Record min/max of    │
  │               │     │  each activation      │
  └──────────────┘     └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  Compute scale (s)    │
                       │  and zero-point (z)   │
                       │  for each layer       │
                       └──────────────────────┘
```

**Calibration methods:**

| Method | Formula | Quality | Speed |
|--------|---------|---------|-------|
| MinMax | $s = (x_{\max} - x_{\min})/255$ | Good | Fastest |
| Percentile | $s = (P_{99.99} - P_{0.01})/255$ | Better | Fast |
| Entropy (KL) | Minimize $\text{KL}(P_{\text{fp32}} \| P_{\text{int8}})$ | Best | Slower |

### Quantization Error Analysis

The quantization error per value:

$$
\epsilon = \lvert x - \hat{x} \rvert \leq \frac{s}{2}
$$

For a layer output $y = Wx$:

$$
\lVert y_{\text{fp32}} - y_{\text{int8}} \rVert \leq \lVert W \rVert_F \cdot \frac{s_x}{2} + \lVert x \rVert \cdot \frac{s_W}{2}
$$

---

## Technique 3: Knowledge Distillation

### Idea — Compress 7B → 1B

Train a small "student" model to mimic a large "teacher" model:

```
  Teacher (7B, frozen)                Student (1B, training)
  ┌──────────────┐                    ┌──────────────┐
  │              │                    │              │
  │   Input x ──┤──► z_t (logits)    │   Input x ──┤──► z_s (logits)
  │              │        │           │              │        │
  └──────────────┘        │           └──────────────┘        │
                          │                                    │
                          └───────────┬────────────────────────┘
                                      │
                                      ▼
                            KL(softmax(z_s/T) ‖ softmax(z_t/T))
                                   + CE(z_s, y_true)
```

### Distillation Loss

$$
\mathcal{L}_{\text{KD}} = \alpha T^2 \text{KL}\!\left(\text{softmax}\!\left(\frac{z_s}{T}\right) \middle\| \text{softmax}\!\left(\frac{z_t}{T}\right)\right) + (1-\alpha)\mathcal{L}_{\text{CE}}(z_s, y)
$$

where:
- $z_s, z_t$ = student and teacher logits
- $T$ = temperature (typically 2-20, softens probability distributions)
- $\alpha$ = mixing weight (typically 0.5-0.9)
- $y$ = ground-truth labels

### Why $T^2$ Scaling?

The KL divergence with temperature $T$ involves softmax:

$$
p_i^T = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

The gradient of KL with temperature:

$$
\frac{\partial \text{KL}}{\partial z_s} = \frac{1}{T}(p_s^T - p_t^T)
$$

This is scaled down by $1/T$ compared to standard CE. The $T^2$ factor compensates:

$$
T^2 \cdot \frac{1}{T}(p_s^T - p_t^T) = T(p_s^T - p_t^T)
$$

ensuring the distillation gradient has the right magnitude relative to the CE gradient.

### Temperature Effect on "Dark Knowledge"

| $T$ | Softmax Output | What Student Learns |
|-----|---------------|-------------------|
| 1 | Sharp (near one-hot) | Only the top prediction |
| 4 | Softer | Relative ranking of top classes |
| 10 | Very soft | Full inter-class relationships |
| 20 | Near-uniform | Subtle similarities between all classes |

---

## Technique 4: `torch.compile` — 1-Line 2x Speedup

### What It Does

```python
model = torch.compile(model)  # That's it!
```

Under the hood, `torch.compile` applies a 3-stage pipeline:

```
  Python Model
       │
       ▼  TorchDynamo
  ┌──────────────────┐
  │ FX Graph capture  │  Traces Python code into a graph IR
  │ (handles control  │  Handles if/else, loops, etc.
  │  flow!)           │
  └────────┬─────────┘
           │
       ▼  AOTAutograd
  ┌──────────────────┐
  │ Ahead-of-Time    │  Pre-computes backward pass graph
  │ Autograd          │  Enables cross-op optimizations
  └────────┬─────────┘
           │
       ▼  Inductor (default backend)
  ┌──────────────────┐
  │ Code generation   │  Generates Triton kernels (GPU)
  │ Triton / C++     │  or C++ code (CPU)
  │ Kernel fusion    │  Fuses elementwise + reduction ops
  └──────────────────┘
```

### Key Optimizations

| Optimization | What | Typical Speedup |
|-------------|------|----------------|
| Kernel fusion | Merge GELU+Linear into single kernel | 10-30% |
| Memory planning | Reuse tensor memory across ops | 5-15% |
| Triton codegen | Custom GPU kernels (not cuDNN) | 10-50% |
| Graph breaks elimination | Minimize Python fallbacks | Variable |

### Compilation Modes

| Mode | Quality | Compile Time | Speedup |
|------|---------|-------------|---------|
| `default` | Balanced | ~30s | 1.5-2x |
| `reduce-overhead` | Fewer graph breaks | ~60s | 1.7-2.2x |
| `max-autotune` | Best kernels | ~5min | 2-3x |

---

## Comprehensive Speed & Size Comparison

| Method | Model Size | Latency | Throughput | Quality | Effort |
|--------|-----------|---------|-----------|---------|--------|
| FP32 baseline | 100% | 1.0x | 1.0x | 100% | None |
| `torch.compile` | 100% | ~0.55x | ~1.8x | ~100% | 1 line |
| ONNX Runtime (fp32) | 100% | ~0.5x | ~2.0x | ~100% | Low |
| ONNX + INT8 PTQ | 25% | ~0.35x | ~3.0x | 97-99% | Medium |
| TensorRT (fp16) | 50% | ~0.25x | ~4.0x | ~99.5% | Medium |
| KD (student) | ~25% | ~0.3x | ~3.5x | ~95% | High |
| KD + INT8 | ~6% | ~0.15x | ~7.0x | ~93% | High |

---

## Benchmarking — How to Measure

### Latency Measurement

```python
# Warm up (first run includes compilation/caching)
for _ in range(10):
    model(dummy_input)

# Measure
times = []
for _ in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    model(input)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

print(f"p50: {np.percentile(times, 50):.1f}ms")
print(f"p99: {np.percentile(times, 99):.1f}ms")
```

### Throughput Measurement

$$
\text{Throughput} = \frac{\text{Batch size}}{\text{Latency per batch}} \quad \text{(samples/second)}
$$

### Memory Measurement

$$
\text{Peak Memory} = \text{Model params} + \text{Activations} + \text{Framework overhead}
$$

---

## Production Deployment Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                 Production Stack                         │
  │                                                          │
  │  Client ──► Load Balancer ──► API Server (FastAPI)      │
  │                                    │                     │
  │                                    ▼                     │
  │                            ┌──────────────┐              │
  │                            │  Model Server │              │
  │                            │              │              │
  │                            │  Option A:   │              │
  │                            │  ONNX Runtime│              │
  │                            │  (CPU/GPU)   │              │
  │                            │              │              │
  │                            │  Option B:   │              │
  │                            │  TorchServe  │              │
  │                            │  + compiled  │              │
  │                            │              │              │
  │                            │  Option C:   │              │
  │                            │  vLLM (VLMs) │              │
  │                            │  + cont.batch│              │
  │                            └──────────────┘              │
  │                                                          │
  │  Monitoring: Prometheus + Grafana                        │
  │  Metrics: latency p50/p99, throughput, GPU util, errors  │
  └─────────────────────────────────────────────────────────┘
```

---

## Congratulations!

This is the **final notebook** of the course. You've built, trained, aligned, and deployed multimodal AI models from scratch.

**Your journey:**

```
  Module 00: Setup
       │
       ▼
  Module 01: Foundations (embeddings, encoders, fusion)
       │
       ▼
  Module 02: Vision-Language Models (CLIP, captioning, VQA)
       │
       ▼
  Module 03: Training (contrastive, multi-objective, SFT→RL→Merge)
       │
       ▼
  Module 04: Efficient Finetuning (LoRA, QLoRA, adapters)
       │
       ▼
  Module 05: Advanced (LLaVA, audio/video, deployment)  ← YOU ARE HERE
```

---

## What You'll Build

- ONNX export of a CLIP model
- INT8 quantization with calibration
- Knowledge distillation: train a small student
- `torch.compile` speed benchmarks
- End-to-end latency and throughput measurement

---

## Prerequisites

- Module 04 (LoRA, quantization concepts from QLoRA)
- Module 05 Notebooks 01–02 (LLaVA, multi-modal models)
- Basic understanding of model serving and APIs
