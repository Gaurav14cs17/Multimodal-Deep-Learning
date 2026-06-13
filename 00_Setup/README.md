# Module 00: Setup & Environment Check

> **Time:** ~15 minutes | **Notebooks:** 1 | **Difficulty:** Beginner

| Notebook | Open in Colab |
|----------|---------------|
| `00_environment_check/00_environment_check.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/00_Setup/00_environment_check/00_environment_check.ipynb) |

---

## Overview

Before we build any models, let's make sure your machine is ready. This module runs a quick health check on your Python environment, detects your hardware (CPU/GPU), and benchmarks your compute speed — so you know what to expect throughout the course.

By the end of this module, you'll have:

- Verified all 20+ dependencies are installed correctly
- Benchmarked your hardware performance
- A visual roadmap of the entire 17-notebook learning journey

---

## What's Inside

### `00_environment_check/00_environment_check.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/00_Setup/00_environment_check/00_environment_check.ipynb)

This single notebook does everything you need to get started.

---

### Part 1: Dependency Verification

The notebook checks every library in `requirements.txt` and displays a clean status table:

```
+-------------------+----------+--------+
| Library           | Required | Status |
+-------------------+----------+--------+
| torch             | >= 2.0   |   OK   |
| transformers      | >= 4.40  |   OK   |
| peft              | >= 0.11  |   OK   |
| bitsandbytes      | >= 0.43  |   OK   |
| matplotlib        | >= 3.8   |   OK   |
| ...               | ...      |  ...   |
+-------------------+----------+--------+
```

If anything is missing, you get a copy-pasteable `pip install` command.

---

### Part 2: Hardware Detection

```
+-----------------------------+
|     YOUR COMPUTE PROFILE    |
+-----------------------------+
| Python:  3.11.5             |
| Device:  CPU / CUDA / MPS   |
| GPU:     (if available)     |
| VRAM:    (if GPU detected)  |
| RAM:     16 GB              |
| Cores:   8                  |
+-----------------------------+
```

All notebooks in this course run on **CPU**. A GPU just makes training faster — it's never required.

---

### Part 3: Performance Benchmark

The notebook times matrix multiplications at increasing sizes to measure your hardware's throughput.

#### The Math Behind the Benchmark

For an $n \times n$ matrix multiplication $C = AB$:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

Total floating-point operations (FLOPs):

$$\text{FLOPs} = 2n^3$$

(each of the $n^2$ output elements requires $n$ multiplications and $n-1$ additions $\approx 2n$ ops)

**GFLOPS** (billions of FLOPs per second):

$$\text{GFLOPS} = \frac{2n^3}{t \times 10^9}$$

where $t$ is latency in seconds.

#### Expected Results

| Matrix Size | FLOPs | Typical CPU Latency | GFLOPS |
|------------|-------|---------------------|--------|
| $128^2$ | $4.2 \times 10^6$ | ~0.3 ms | ~14 |
| $256^2$ | $3.4 \times 10^7$ | ~0.8 ms | ~42 |
| $512^2$ | $2.7 \times 10^8$ | ~3.1 ms | ~86 |
| $1024^2$ | $2.1 \times 10^9$ | ~19 ms | ~114 |
| $2048^2$ | $1.7 \times 10^{10}$ | ~142 ms | ~121 |

Note: GFLOPS increases with matrix size because larger matrices better utilize CPU caches and SIMD vector units (AVX2/AVX-512).

---

### Part 4: Course Roadmap

A color-coded visual diagram showing all 6 modules and how they connect:

```
  +----------+      +--------------+      +----------------+
  | Module 00|      | Module 01    |      | Module 02      |
  | Setup    | ---> | Foundations   | ---> | Vision-Language|
  | (15 min) |      | (1-2 hrs)    |      | (2-3 hrs)      |
  +----------+      +--------------+      +----------------+
                                                  |
                                                  v
  +----------+      +--------------+      +----------------+
  | Module 05|      | Module 04    |      | Module 03      |
  | Advanced | <--- | Finetuning   | <--- | Training       |
  | (2-3 hrs)|      | (3-4 hrs)    |      | (2-3 hrs)      |
  +----------+      +--------------+      +----------------+
                        CORE              CORE
```

---

## Prerequisites

```bash
pip install -r ../requirements.txt
```

That's it. No GPUs, no data downloads, no API keys.

---

## Checklist Before Moving On

- [ ] All library imports succeed without errors
- [ ] Hardware is detected (CPU is perfectly fine)
- [ ] Benchmark completes without crashes
- [ ] You've seen the learning roadmap and know what's ahead

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r ../requirements.txt` |
| `CUDA not available` | That's OK — all notebooks work on CPU |
| Benchmark crashes | Try reducing max matrix size to 1024 |
| Jupyter won't start | Run `pip install jupyterlab>=4.0` |

---

## Next Step

You're all set! Head to **Module 01** to start building:

**[01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb](../01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb)
