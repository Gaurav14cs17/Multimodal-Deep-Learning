# 00 — Environment Check

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/00_Setup/00_environment_check/00_environment_check.ipynb)

> **Time:** ~2 minutes | **Difficulty:** Beginner | **GPU Required:** No

---

## What This Notebook Does

Verifies that your environment has everything needed to run the course. It performs a systematic check of all required libraries, hardware, and utilities.

---

## System Requirements

| Component | Minimum | Recommended | What It's Used For |
|-----------|---------|-------------|-------------------|
| Python | 3.8+ | 3.10+ | Language runtime |
| PyTorch | 1.12+ | 2.0+ | Deep learning framework |
| Transformers | 4.28+ | 4.36+ | HuggingFace models |
| PEFT | 0.4+ | 0.7+ | LoRA, adapters |
| Datasets | 2.0+ | 2.14+ | Data loading |
| matplotlib | 3.5+ | 3.8+ | Visualization |
| Pillow (PIL) | 8.0+ | 10.0+ | Image processing |
| GPU (optional) | 4 GB VRAM | 16+ GB VRAM | Accelerated training |

---

## Diagnostic Flow

```
  ┌───────────────────────────────────────────┐
  │          Environment Check Flow            │
  │                                            │
  │  Step 1: Python Version                    │
  │  ├── ≥ 3.8? ✓ Continue                    │
  │  └── < 3.8? ✗ Upgrade Python              │
  │                                            │
  │  Step 2: Core Libraries                    │
  │  ├── torch?        → Version + CUDA info   │
  │  ├── transformers? → Version               │
  │  ├── peft?         → Version               │
  │  ├── datasets?     → Version               │
  │  └── Missing?      → pip install -r ...    │
  │                                            │
  │  Step 3: Compute Device                    │
  │  ├── CUDA GPU?  → Report name + VRAM       │
  │  ├── Apple MPS? → Report availability      │
  │  └── CPU only?  → OK (slower but works)    │
  │                                            │
  │  Step 4: Utility Modules                   │
  │  ├── utils/ importable? → ✓               │
  │  └── Import error?      → Check sys.path  │
  │                                            │
  │  Result: ✓ All checks passed / ✗ Fix needed│
  └───────────────────────────────────────────┘
```

---

## What You'll See

| Check | What It Verifies | Typical Output |
|-------|-----------------|----------------|
| Python | Version $\geq$ 3.8 | `Python 3.10.12` |
| PyTorch | Installed + CUDA status | `torch 2.1.0+cu121` |
| Transformers | HuggingFace library | `transformers 4.36.0` |
| PEFT | Parameter-efficient FT | `peft 0.7.1` |
| Device | CPU / GPU / MPS | `cuda:0 (NVIDIA A100, 40GB)` |
| Memory | Available VRAM (if GPU) | `39.6 GB free / 40.0 GB total` |

---

## Quick Start

```bash
# Local
cd 00_Setup/00_environment_check
jupyter notebook 00_environment_check.ipynb

# Or just click the Colab badge above
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` from the repo root |
| No GPU detected | Normal — all notebooks run on CPU (slower but functional) |
| CUDA version mismatch | Install PyTorch matching your CUDA: [pytorch.org](https://pytorch.org/get-started/locally/) |
| `ImportError: utils` | Run from the repo root directory, or add it to `sys.path` |
| Colab disconnects | Enable GPU: Runtime → Change runtime type → T4 GPU |
| Low VRAM warning | Some notebooks need $\geq$ 8GB; use Colab for free T4 (16GB) |

---

## Next Step

**[01_what_is_multimodal](../../01_Multimodal_Foundations/01_what_is_multimodal/)** — What is multimodal learning and why does it matter?
