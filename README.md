# Multimodal Deep Learning - A Complete Practical Guide

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Notebooks](https://img.shields.io/badge/Notebooks-17-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Low Compute](https://img.shields.io/badge/Runs%20On-CPU%20%2F%20Free%20Colab-brightgreen)

> **Master multimodal AI step-by-step with 17 visual, hands-on Jupyter notebooks.**
> Designed for **low compute** environments (CPU / single GPU / Google Colab free tier).

---

## Table of Contents

- [Why This Project?](#why-this-project)
- [Who Is This For?](#who-is-this-for)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Learning Path](#learning-path-recommended-order)
- [What You'll Build](#what-youll-build-from-scratch)
- [Hardware Requirements](#hardware-requirements)
- [Key Design Principles](#key-design-principles)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [What's Next After This Course?](#whats-next-after-this-course)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)
- [Version History](#version-history)

---

## Why This Project?

Multimodal AI (combining images, text, audio, video) powers the most exciting models today: **GPT-4V, Gemini, LLaVA, CLIP, DALL-E**. But most tutorials either:
- Show only theory with no runnable code
- Require expensive hardware (A100 GPUs, 80GB VRAM)
- Skip the "how to actually train" part

**This project is different.** Every concept is:
1. **Built from scratch** in PyTorch (you see every line)
2. **Visualized** with diagrams, heatmaps, and charts (54 visual plots total)
3. **Runnable on CPU** with small models and synthetic data (no downloads needed)
4. **Focused on training & finetuning** — the skills you actually need

---

## Who Is This For?

You should already know:
- Self-Attention, Multi-Head Attention, Cross-Attention
- Transformer architecture (encoder, decoder)
- Basic PyTorch (tensors, nn.Module, training loops)

You will learn:
- How multimodal models work internally
- How to train them from scratch
- How to finetune large models with LoRA/QLoRA on limited hardware
- How to deploy efficiently

---

## Project Structure

```
MultiModel/
│
├── README.md                                    # You are here
├── requirements.txt                             # All dependencies with versions
│
├── 00_Setup/                                    # [15 min]
│   ├── README.md                                # Module guide
│   └── 00_environment_check.ipynb               # Verify setup, benchmark, roadmap
│
├── 01_Multimodal_Foundations/                    # [1-2 hrs]
│   ├── README.md                                # Module guide
│   ├── 01_what_is_multimodal.ipynb              # Landscape, alignment problem
│   ├── 02_modality_encoders.ipynb               # ViT & text encoder from scratch
│   └── 03_fusion_strategies.ipynb               # Early/Late/Cross/Gated fusion
│
├── 02_Vision_Language_Models/                   # [2-3 hrs]
│   ├── README.md                                # Module guide
│   ├── 01_clip_from_scratch.ipynb               # Build & train CLIP end-to-end
│   ├── 02_image_captioning.ipynb                # Encoder-decoder captioning
│   └── 03_visual_question_answering.ipynb       # VQA with attention maps
│
├── 03_Training_Strategies/                      # [2-3 hrs] ★ CORE FOCUS
│   ├── README.md                                # Module guide
│   ├── 01_contrastive_learning.ipynb            # InfoNCE deep dive + temperature
│   ├── 02_pretraining_objectives.ipynb          # ITC, ITM, MLM, generation
│   └── 03_training_pipeline.ipynb               # Full loop, grad accum, scheduling
│
├── 04_Finetuning_LowCompute/                   # [3-4 hrs] ★ CORE FOCUS
│   ├── README.md                                # Module guide
│   ├── 01_lora_from_scratch.ipynb               # Build LoRA, rank analysis
│   ├── 02_qlora_4bit_finetuning.ipynb           # NF4 quantization + QLoRA
│   ├── 03_adapter_methods.ipynb                 # Prefix/Prompt/IA3/BitFit
│   └── 04_finetune_clip_custom_data.ipynb       # End-to-end LoRA finetuning
│
├── 05_Advanced_Topics/                          # [2-3 hrs]
│   ├── README.md                                # Module guide
│   ├── 01_llava_architecture.ipynb              # Mini-LLaVA from scratch
│   ├── 02_multimodal_beyond_vision.ipynb        # Audio + Video encoders
│   └── 03_efficient_deployment.ipynb            # ONNX, quantization, Gradio
│
├── utils/                                       # Shared code
│   ├── README.md                                # API reference
│   ├── __init__.py
│   ├── visualization.py                         # 10+ plotting functions
│   └── helpers.py                               # Training loop, data utils
│
└── assets/                                      # Auto-generated diagrams (created on first run)
```

---

## Quick Start

### Option 1: Local Machine

```bash
# Clone or download this project
cd MultiModel

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab

# Open: 00_Setup/00_environment_check.ipynb
```

### Option 2: Google Colab (Zero Setup — Click & Run)

Every notebook has a built-in Colab setup cell that clones the repo and installs dependencies automatically. Just click any badge below:

| # | Notebook | Open in Colab |
|---|----------|---------------|
| 00 | Environment Check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/00_Setup/00_environment_check.ipynb) |
| 01 | What is Multimodal? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/01_what_is_multimodal.ipynb) |
| 02 | Modality Encoders | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/02_modality_encoders.ipynb) |
| 03 | Fusion Strategies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/03_fusion_strategies.ipynb) |
| 04 | CLIP from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/01_clip_from_scratch.ipynb) |
| 05 | Image Captioning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/02_image_captioning.ipynb) |
| 06 | Visual Question Answering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/03_visual_question_answering.ipynb) |
| 07 | Contrastive Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning.ipynb) |
| 08 | Pretraining Objectives | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives.ipynb) |
| 09 | Training Pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline.ipynb) |
| 10 | LoRA from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/01_lora_from_scratch.ipynb) |
| 11 | QLoRA 4-bit Finetuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/02_qlora_4bit_finetuning.ipynb) |
| 12 | Adapter Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/03_adapter_methods.ipynb) |
| 13 | Finetune CLIP with LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/04_finetune_clip_custom_data.ipynb) |
| 14 | LLaVA Architecture | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/01_llava_architecture.ipynb) |
| 15 | Multimodal Beyond Vision | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/02_multimodal_beyond_vision.ipynb) |
| 16 | Efficient Deployment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/03_efficient_deployment.ipynb) |

---

## Learning Path (Recommended Order)

```
Module 00 ──→ Module 01 ──→ Module 02 ──→ Module 03 ──→ Module 04 ──→ Module 05
 (Setup)     (Foundations)   (Models)     (Training)    (Finetuning)   (Advanced)
 15 min       1-2 hrs       2-3 hrs       2-3 hrs       3-4 hrs       2-3 hrs
                                            ★               ★
                                        CORE FOCUS      CORE FOCUS
```

### Detailed Module Breakdown

| # | Module | Notebooks | What You Build | Key Visuals |
|---|--------|-----------|---------------|-------------|
| **00** | Setup | 1 | Environment check | Benchmark charts, roadmap |
| **01** | Foundations | 3 | ViT encoder, text encoder, fusion layers | Architecture diagrams, patch visualization, encoder comparison |
| **02** | Vision-Language | 3 | CLIP, image captioner, VQA model | Similarity matrices, attention heatmaps, generation steps |
| **03** | Training | 3 | Contrastive loss, multi-objective, full pipeline | InfoNCE step-by-step, temperature analysis, training evolution |
| **04** | Finetuning | 4 | LoRA layer, QLoRA, adapters, CLIP finetune | LoRA diagram, memory comparison, parameter savings |
| **05** | Advanced | 3 | Mini-LLaVA, audio/video encoder, ONNX export | LLaVA architecture, deployment options, speed benchmarks |
| | **Total** | **17** | | **54 visual plots** |

---

## What You'll Build (From Scratch)

### Models Built in This Course

| Model | Notebook | Parameters | Runs On |
|-------|----------|-----------|---------|
| **ViT (Vision Transformer)** | 01.02 | ~500K | CPU |
| **Text Encoder (BERT-style)** | 01.02 | ~500K | CPU |
| **Early/Late/Cross-Modal Fusion** | 01.03 | ~200K each | CPU |
| **Mini-CLIP** | 02.01 | ~1.5M | CPU |
| **Image Captioning Model** | 02.02 | ~1.2M | CPU |
| **VQA Model** | 02.03 | ~1M | CPU |
| **LoRA Layer** | 04.01 | Custom | CPU |
| **Three-Modal CLIP (Img+Txt+Audio)** | 05.02 | ~800K | CPU |
| **Mini-LLaVA** | 05.01 | ~2M | CPU |
| **Video Encoder** | 05.02 | ~300K | CPU |

### Techniques Covered

| Technique | Where | Low Compute? |
|-----------|-------|-------------|
| Contrastive Learning (InfoNCE) | 03.01 | Yes |
| Multi-objective Pretraining (ITC+ITM+MLM) | 03.02 | Yes |
| Gradient Accumulation | 03.03 | Yes (essential!) |
| Mixed Precision (fp16) | 03.03 | Yes |
| Cosine LR with Warmup | 03.03 | Yes |
| **LoRA** | 04.01 | Yes |
| **QLoRA (4-bit NF4)** | 04.02 | Yes |
| Prefix Tuning | 04.03 | Yes |
| Prompt Tuning | 04.03 | Yes |
| IA3 | 04.03 | Yes |
| Adapters (Bottleneck) | 04.03 | Yes |
| ONNX Export | 05.03 | Yes |
| Post-training Quantization (INT8) | 05.03 | Yes |
| torch.compile | 05.03 | Yes |

---

## Core Mathematical Concepts

This course covers the complete mathematical pipeline from raw inputs to trained multimodal models. Here are the key equations you'll master:

### Embedding & Attention

| Concept | Equation | Module |
|---------|----------|--------|
| **Patch Embedding** | $\mathbf{z}_i = \mathbf{x}_p^i \cdot \mathbf{E} + \mathbf{e}_{\text{pos}}^i$ | 01 |
| **Self-Attention** | $\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | 01 |
| **Cross-Attention** | $Q = \mathbf{Z}_{\text{txt}} W_Q, \; K,V = \mathbf{Z}_{\text{img}} W_{K,V}$ | 01, 02 |
| **Cosine Similarity** | $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ | 02 |

### Loss Functions

| Loss | Equation | Module |
|------|----------|--------|
| **InfoNCE (CLIP)** | $\mathcal{L} = -\frac{1}{N}\sum_i \log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$ | 02, 03 |
| **Captioning** | $\mathcal{L} = -\sum_t \log P(y_t \mid y_{<t}, I)$ | 02 |
| **VQA** | $\mathcal{L} = -\log P(a \mid I, Q)$ | 02 |

### Parameter-Efficient Finetuning

| Method | Equation | Savings | Module |
|--------|----------|---------|--------|
| **LoRA** | $h = W_0 x + \frac{\alpha}{r} BAx$ | ~98% | 04 |
| **QLoRA** | $h = \text{Dequant}(W_{\text{NF4}}) x + BAx$ | ~99.7% | 04 |
| **IA3** | $K' = \mathbf{l}_K \odot K, \; V' = \mathbf{l}_V \odot V$ | ~99.6% | 04 |

### Training Tricks

| Technique | Formula | Module |
|-----------|---------|--------|
| **Gradient Accumulation** | $\bar{g} = \frac{1}{K}\sum_k \nabla\mathcal{L}(\mathcal{B}_k)$ | 03 |
| **Cosine LR + Warmup** | $\eta(t) = \frac{\eta_{\max}}{2}\left(1 + \cos\frac{\pi t}{T}\right)$ | 03 |
| **Gradient Clipping** | $\hat{g} = c \cdot g / \|g\|$ if $\|g\| > c$ | 03 |

---

## Hardware Requirements

| Setup | Works? | Notes |
|-------|--------|-------|
| **CPU only (no GPU)** | Yes | All notebooks designed for this |
| **Google Colab Free (T4)** | Yes | Best free option |
| **Laptop GPU (4-8 GB)** | Yes | Faster training |
| **Desktop GPU (12+ GB)** | Yes | Can run pretrained models too |

### Memory Usage Per Notebook

All from-scratch models use **< 500 MB RAM**. Pretrained model cells (commented out) need:
- BLIP captioning: ~1 GB download
- OpenCLIP: ~600 MB download
- QLoRA (7B model): ~4 GB GPU (with NF4)

---

## Key Design Principles

### 1. Visual First
Every concept has a programmatic diagram. No external images — all visuals are generated in code so you can modify and re-run them.

### 2. From Scratch Before Libraries
We build LoRA, CLIP, ViT, attention, and fusion layers from raw PyTorch before showing the PEFT/HuggingFace library versions. You understand what happens inside.

### 3. Low Compute by Default
- Models are small (128-256 dim, 2-4 layers)
- Data is synthetic (no downloads)
- Training takes seconds, not hours
- GPU code is always optional

### 4. Progressive Complexity
Each notebook builds on the previous. Module 01 teaches encoders, Module 02 uses them to build CLIP, Module 03 teaches how to train CLIP, Module 04 teaches how to finetune it cheaply.

---

## Dependencies

All versions are pinned in `requirements.txt`. Core libraries:

| Library | Purpose | Version |
|---------|---------|---------|
| `torch` | Deep learning framework | >= 2.0 |
| `transformers` | Pretrained models | >= 4.40 |
| `peft` | LoRA, QLoRA, adapters | >= 0.11 |
| `bitsandbytes` | 4-bit quantization | >= 0.43 |
| `datasets` | HuggingFace datasets | >= 2.19 |
| `matplotlib` | All visualizations | >= 3.8 |
| `seaborn` | Heatmaps | >= 0.13 |
| `einops` | Tensor operations | >= 0.7 |
| `open-clip-torch` | Pretrained CLIP | >= 2.24 |
| `gradio` | Demo apps | >= 4.0 |
| `onnxruntime` | ONNX inference | >= 1.18 |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `CUDA out of memory` | All notebooks work on CPU — just use `device='cpu'` |
| Notebook shows no output | Make sure to **Run All Cells** in order |
| `utils` import fails | Run notebooks from their own folder, not the project root |
| Slow training | This is expected on CPU — models are small so it finishes in seconds |

---

## What's Next After This Course?

1. **Finetune CLIP** on your own domain images (use notebook 04.04 as template)
2. **Build a LLaVA chatbot** with QLoRA on a 7B LLM
3. **Deploy with Gradio** — use the template in notebook 05.03
4. **Explore HuggingFace Hub** for pretrained multimodal models
5. **Read the papers**: CLIP, LLaVA, BLIP-2, QLoRA, LoRA

### Recommended Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | Contrastive image-text pretraining |
| [ViT](https://arxiv.org/abs/2010.11929) | 2020 | Vision Transformer |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | Low-rank adaptation |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 2023 | 4-bit quantized LoRA |
| [LLaVA](https://arxiv.org/abs/2304.08485) | 2023 | Visual instruction tuning |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 2023 | Bootstrapped vision-language |
| [ImageBind](https://arxiv.org/abs/2305.05665) | 2023 | 6-modality binding |

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs** — Open an issue if a notebook has errors or doesn't run
2. **Improve explanations** — Submit a PR if a concept could be explained better
3. **Add examples** — Create additional notebooks that extend the existing modules
4. **Fix typos** — Small improvements are valued too

### How to Contribute

```bash
# Fork this repository
git clone https://github.com/<your-username>/MultiModel.git
cd MultiModel

# Create a feature branch
git checkout -b feature/your-improvement

# Make your changes and test them
jupyter lab  # verify notebooks run end-to-end

# Commit and push
git add .
git commit -m "Describe your change"
git push origin feature/your-improvement

# Open a Pull Request
```

**Guidelines:**
- All notebooks should run on **CPU without downloads** (use synthetic data)
- Follow the existing naming convention (`XX_descriptive_name.ipynb`)
- Include at least one visualization per major concept
- Test on Python 3.9+ with the pinned dependencies

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to:
- Use this material for personal learning
- Adapt it for courses or workshops (with attribution)
- Fork and modify for your own projects

---

## Author

**Gaurav Goswami**

- GitHub: [@ggoswami](https://github.com/ggoswami)

If you find this project helpful, consider giving it a star on GitHub!

---

## Acknowledgments

This project draws inspiration from and builds upon the work of:

| Resource | Credit |
|----------|--------|
| [CLIP (OpenAI)](https://openai.com/research/clip) | Architecture and contrastive learning approach |
| [LLaVA](https://llava-vl.github.io/) | Visual instruction tuning framework |
| [LoRA](https://arxiv.org/abs/2106.09685) | Parameter-efficient finetuning |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 4-bit quantized finetuning |
| [HuggingFace](https://huggingface.co/) | Transformers, PEFT, Datasets libraries |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [Andrej Karpathy](https://karpathy.ai/) | Educational approach to building from scratch |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.0.0** | June 2026 | Initial release — 17 notebooks, 6 modules, 54 visualizations |

---

**Total: 17 notebooks | 104 code cells | 79 markdown cells | 54 visual plots**

**Estimated total time: 12-16 hours of focused learning.**

Happy learning!
