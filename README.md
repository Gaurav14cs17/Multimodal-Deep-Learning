# Multimodal Deep Learning — A Complete Practical Guide

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Notebooks](https://img.shields.io/badge/Notebooks-28-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Low Compute](https://img.shields.io/badge/Runs%20On-CPU%20%2F%20Free%20Colab-brightgreen)

> **Master multimodal AI step-by-step with 28 visual, hands-on Jupyter notebooks.**
> Designed for **low compute** environments (CPU / single GPU / Google Colab free tier).

```
           ┌──────────┐    ┌──────────┐    ┌──────────┐
           │  Image   │    │   Text   │    │  Audio   │
           │  Input   │    │  Input   │    │  Input   │
           └────┬─────┘    └────┬─────┘    └────┬─────┘
                │               │               │
                ▼               ▼               ▼
          ┌──────────┐    ┌──────────┐    ┌──────────┐
          │   ViT    │    │  BERT    │    │   Mel    │
          │ Encoder  │    │ Encoder  │    │ Encoder  │
          └────┬─────┘    └────┬─────┘    └────┬─────┘
                │               │               │
                ▼               ▼               ▼
           ┌─────────────────────────────────────────┐
           │        Shared Embedding Space ℝᴰ        │
           │                                         │
           │   sim(img, txt) = v̂ᵀ · t̂  ∈ [-1, 1]   │
           └──────────────────┬──────────────────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │ Alignment│  │  Fusion  │  │Generation│
         │  (CLIP)  │  │  (VQA)   │  │(Caption) │
         └──────────┘  └──────────┘  └──────────┘
```

---

## Table of Contents

- [Why This Project?](#why-this-project)
- [Who Is This For?](#who-is-this-for)
- [The Multimodal Pipeline — Big Picture](#the-multimodal-pipeline--big-picture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Learning Path](#learning-path-recommended-order)
- [What You'll Build](#what-youll-build-from-scratch)
- [Core Mathematical Concepts](#core-mathematical-concepts)
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
4. **Mathematically rigorous** — full derivations from dot product to InfoNCE to LoRA

---

## Who Is This For?

**You should already know:**
- Self-Attention, Multi-Head Attention, Cross-Attention
- Transformer architecture (encoder, decoder)
- Basic PyTorch (tensors, `nn.Module`, training loops)

**You will learn:**
- How multimodal models work internally — with every equation derived
- How to train them from scratch (contrastive, generative, discriminative)
- How to finetune large models with LoRA/QLoRA on limited hardware
- How to deploy efficiently (ONNX, INT8, torch.compile)

---

## The Multimodal Pipeline — Big Picture

Every multimodal model follows the same three-stage pattern. This course teaches you each stage in depth:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE MULTIMODAL PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: ENCODE              STAGE 2: ALIGN / FUSE        STAGE 3: TASK   │
│  (Module 01-02)               (Module 02-03)               (Module 04-05)  │
│                                                                             │
│  Image ──→ ViT ──→ v ∈ ℝᴰ ─┐                                              │
│                              ├──→ Contrastive ──→ CLIP     (alignment)     │
│  Text ──→ BERT ──→ t ∈ ℝᴰ ─┘    (InfoNCE)                                │
│                              ├──→ Cross-Attn ──→ VQA       (classification)│
│  Audio ──→ Mel+ViT ──→ a ∈ ℝᴰ   (Q from text)                            │
│                              └──→ Enc-Dec ──→ Caption      (generation)    │
│                                                                             │
│  Key equation per stage:                                                    │
│                                                                             │
│  Encode:  h = LN(MSA(Z) + Z),  then h = LN(FFN(h) + h)                   │
│  Align:   L = -log [exp(sᵢᵢ/τ) / Σⱼ exp(sᵢⱼ/τ)]                         │
│  Finetune: h = W₀x + (α/r)·BA·x    (LoRA — 98% fewer params)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Evolution of Multimodal AI

```
 2020          2021          2022          2023          2024          2025+
  │             │             │             │             │             │
  ViT          CLIP          BLIP         LLaVA         GPT-4V       Gemini 2
  │            ALIGN         BLIP-2       Gemini        LLaVA-1.5    Qwen2-VL
  │             │           Flamingo     ImageBind      Qwen-VL      LLaVA-Next
  │             │             │             │             │             │
  ▼             ▼             ▼             ▼             ▼             ▼
[Vision       [V+L          [V+L          [V+L          [Omni-        [Agent
 only]        Align]        Generate]     +LLM]         modal]        Vision]

  Attention → Contrastive → Cross-Attn → Instruction → Unified → Tool Use
  is all      learning      fusion       tuning        perception
  you need
```

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
│   └── 00_environment_check/00_environment_check.ipynb               # Verify setup, benchmark, roadmap
│
├── 01_Multimodal_Foundations/                    # [2-3 hrs]
│   ├── README.md                                # Deep math guide: ViT, attention, fusion
│   ├── 01_what_is_multimodal/01_what_is_multimodal.ipynb              # Landscape, alignment problem
│   ├── 02_modality_encoders/02_modality_encoders.ipynb               # ViT & text encoder from scratch
│   ├── 03_fusion_strategies/03_fusion_strategies.ipynb               # Early/Late/Cross/Gated fusion
│   ├── 04_attention_mechanism/04_attention_mechanism.ipynb           # Self-attn, multi-head, complexity
│   ├── 05_transformer_architecture/05_transformer_architecture.ipynb # Full encoder-decoder
│   └── 06_tokenization_embeddings/06_tokenization_embeddings.ipynb   # BPE, WordPiece, patch embed
│
├── 02_Vision_Language_Models/                   # [2-3 hrs]
│   ├── README.md                                # Deep math guide: CLIP, captioning, VQA
│   ├── 01_clip_from_scratch/01_clip_from_scratch.ipynb               # Build & train CLIP end-to-end
│   ├── 02_image_captioning/02_image_captioning.ipynb                # Encoder-decoder captioning
│   └── 03_visual_question_answering/03_visual_question_answering.ipynb       # VQA with attention maps
│
├── 03_Training_Strategies/                      # [3-4 hrs] ★ CORE FOCUS
│   ├── README.md                                # Deep math guide: InfoNCE, AdamW, scheduling
│   ├── 01_contrastive_learning/01_contrastive_learning.ipynb            # InfoNCE deep dive + temperature
│   ├── 02_pretraining_objectives/02_pretraining_objectives.ipynb          # ITC, ITM, MLM, generation
│   ├── 03_training_pipeline/03_training_pipeline.ipynb               # Full loop, grad accum, scheduling
│   ├── 04_multimodal_alignment/04_multimodal_alignment.ipynb           # Projection heads, Recall@K
│   └── 05_scaling_laws/05_scaling_laws.ipynb                         # Chinchilla, compute-optimal
│
├── 04_Finetuning_LowCompute/                   # [3-4 hrs] ★ CORE FOCUS
│   ├── README.md                                # Deep math guide: SVD, LoRA, NF4, PEFT
│   ├── 01_lora_from_scratch/01_lora_from_scratch.ipynb               # Build LoRA, rank analysis
│   ├── 02_qlora_4bit_finetuning/02_qlora_4bit_finetuning.ipynb           # NF4 quantization + QLoRA
│   ├── 03_adapter_methods/03_adapter_methods.ipynb                 # Prefix/Prompt/IA3/BitFit
│   └── 04_finetune_clip_custom_data/04_finetune_clip_custom_data.ipynb       # End-to-end LoRA finetuning
│
├── 05_Advanced_Topics/                          # [4-5 hrs]
│   ├── README.md                                # Deep math guide: LLaVA, STFT, KV cache
│   ├── 01_llava_architecture/01_llava_architecture.ipynb              # Mini-LLaVA from scratch
│   ├── 02_multimodal_beyond_vision/02_multimodal_beyond_vision.ipynb        # Audio + Video encoders
│   ├── 03_efficient_deployment/03_efficient_deployment.ipynb            # ONNX, quantization, Gradio
│   ├── 04_diffusion_models/04_diffusion_models.ipynb                  # DDPM, DDIM, latent diffusion
│   ├── 05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb          # MMMU, MME, MM-Bench
│   ├── 06_multimodal_reasoning/06_multimodal_reasoning.ipynb          # Visual CoT, grounding
│   ├── 07_text_to_image_video/07_text_to_image_video.ipynb            # DALL-E, Stable Diffusion, video
│   └── 08_multimodal_agents/08_multimodal_agents.ipynb                # Tool use, ReAct agents
│
├── 06_Datasets_Benchmarks/                     # [1-2 hrs]
│   ├── README.md                                # Dataset catalog + curation math
│   └── 01_multimodal_datasets/01_multimodal_datasets.ipynb           # Load, analyze, preprocess data
│
├── utils/                                       # Shared code
│   ├── README.md                                # API reference
│   ├── __init__.py
│   ├── visualization.py                         # 10+ plotting functions
│   └── helpers.py                               # Training loop, data utils
│
└── assets/
    └── paper_figures/                           # Real figures from research papers
        ├── clip_overview.png
        ├── vit_architecture.png
        ├── blip2_framework.png
        ├── simclr_framework.png
        ├── llava_arch.png
        ├── lora_diagram.png
        ├── imagebind_overview.png
        ├── rlhf_pipeline.png
        └── mm_cot.png
```

---

## 📄 Paper Figures & References

This project includes real architecture diagrams from landmark research papers:

| Paper | Figure | Used In |
|-------|--------|---------|
| CLIP (Radford et al., 2021) | Contrastive pretraining overview | Module 01, 02 |
| ViT (Dosovitskiy et al., 2020) | Patch + Transformer architecture | Module 01 |
| BLIP-2 (Li et al., 2023) | Q-Former + frozen LLM | Module 02 |
| SimCLR (Chen et al., 2020) | Contrastive framework | Module 03 |
| LLaVA (Liu et al., 2023) | Two-stage training | Module 05 |
| LoRA (Hu et al., 2021) | Low-rank adaptation | Module 04 |
| ImageBind (Girdhar et al., 2023) | 6-modality binding | Module 05 |
| RLHF Pipeline (Ouyang et al., 2022) | SFT → RM → PPO | Module 03 |

### Related Resources

- 📚 [Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) — Curated list of multimodal LLM papers and resources

---

## Quick Start

### Option 1: Local Machine

```bash
# Clone the repository
git clone https://github.com/Gaurav14cs17/Multimodal-Deep-Learning.git
cd Multimodal-Deep-Learning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab

# Open: 00_Setup/00_environment_check/00_environment_check.ipynb
```

### Option 2: Google Colab (Zero Setup — Click & Run)

Every notebook has a built-in Colab setup cell that clones the repo and installs dependencies automatically. Just click any badge below:

| # | Notebook | Open in Colab |
|---|----------|---------------|
| 00 | Environment Check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/00_Setup/00_environment_check/00_environment_check.ipynb) |
| 01 | What is Multimodal? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/01_what_is_multimodal/01_what_is_multimodal.ipynb) |
| 02 | Modality Encoders | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/02_modality_encoders/02_modality_encoders.ipynb) |
| 03 | Fusion Strategies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/03_fusion_strategies/03_fusion_strategies.ipynb) |
| 04 | Attention Mechanism | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/04_attention_mechanism/04_attention_mechanism.ipynb) |
| 05 | Transformer Architecture | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/05_transformer_architecture/05_transformer_architecture.ipynb) |
| 06 | Tokenization & Embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/06_tokenization_embeddings/06_tokenization_embeddings.ipynb) |
| 07 | CLIP from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/01_clip_from_scratch/01_clip_from_scratch.ipynb) |
| 08 | Image Captioning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/02_image_captioning/02_image_captioning.ipynb) |
| 09 | Visual Question Answering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/03_visual_question_answering/03_visual_question_answering.ipynb) |
| 10 | Contrastive Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/01_contrastive_learning/01_contrastive_learning.ipynb) |
| 11 | Pretraining Objectives | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives/02_pretraining_objectives.ipynb) |
| 12 | Training Pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline/03_training_pipeline.ipynb) |
| 13 | Multimodal Alignment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/04_multimodal_alignment/04_multimodal_alignment.ipynb) |
| 14 | Scaling Laws | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/05_scaling_laws/05_scaling_laws.ipynb) |
| 15 | LoRA from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/01_lora_from_scratch/01_lora_from_scratch.ipynb) |
| 16 | QLoRA 4-bit Finetuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/02_qlora_4bit_finetuning/02_qlora_4bit_finetuning.ipynb) |
| 17 | Adapter Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/03_adapter_methods/03_adapter_methods.ipynb) |
| 18 | Finetune CLIP with LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/04_finetune_clip_custom_data/04_finetune_clip_custom_data.ipynb) |
| 19 | LLaVA Architecture | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/01_llava_architecture/01_llava_architecture.ipynb) |
| 20 | Multimodal Beyond Vision | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/02_multimodal_beyond_vision/02_multimodal_beyond_vision.ipynb) |
| 21 | Efficient Deployment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/03_efficient_deployment/03_efficient_deployment.ipynb) |
| 22 | Diffusion Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/04_diffusion_models/04_diffusion_models.ipynb) |
| 23 | Evaluation & Benchmarks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb) |
| 24 | Multimodal Reasoning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/06_multimodal_reasoning/06_multimodal_reasoning.ipynb) |
| 25 | Text-to-Image & Video | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/07_text_to_image_video/07_text_to_image_video.ipynb) |
| 26 | Multimodal Agents | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/08_multimodal_agents/08_multimodal_agents.ipynb) |
| 27 | Multimodal Datasets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/06_Datasets_Benchmarks/01_multimodal_datasets/01_multimodal_datasets.ipynb) |

---

## Learning Path (Recommended Order)

```
Module 00 ──→ Module 01 ──→ Module 02 ──→ Module 03 ──→ Module 04 ──→ Module 05 ──→ Module 06
 (Setup)     (Foundations)   (Models)     (Training)    (Finetuning)   (Advanced)    (Datasets)
 15 min       2-3 hrs       2-3 hrs       3-4 hrs       3-4 hrs       4-5 hrs       1-2 hrs
                                            ★               ★
                                        CORE FOCUS      CORE FOCUS
```

### What Each Module Teaches

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Module 00 ─── "Can my machine run this?"                                   │
│       │         Dependency check, GPU detection, benchmarks                  │
│       ▼                                                                      │
│  Module 01 ─── "How do encoders work?"                                      │
│       │         ViT from scratch, text encoder, fusion strategies            │
│       │         Math: Patch embed, self-attention, √d_k scaling, LayerNorm  │
│       ▼                                                                      │
│  Module 02 ─── "How do I build real models?"                                │
│       │         CLIP, Image Captioner, VQA — all from scratch               │
│       │         Math: InfoNCE, beam search, cross-entropy, zero-shot        │
│       ▼                                                                      │
│  Module 03 ─── "How do I train them properly?"     ★ CORE                   │
│       │         Contrastive learning, multi-objective, full pipeline         │
│       │         Math: MI bound, AdamW, cosine LR, gradient accumulation     │
│       ▼                                                                      │
│  Module 04 ─── "How do I finetune on my laptop?"   ★ CORE                   │
│       │         LoRA, QLoRA, 6 PEFT methods, end-to-end finetuning          │
│       │         Math: SVD, Eckart-Young, NF4 quantiles, rank selection      │
│       ▼                                                                      │
│  Module 05 ─── "How do I build & deploy LLaVA?"                             │
│       │         LLaVA, diffusion, eval, reasoning, agents, deployment       │
│       ▼                                                                      │
│  Module 06 ─── "What data do I train on?"                                   │
│               LAION/CC/DataComp, instruction data, benchmarks, curation     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Module Breakdown

| # | Module | Notebooks | What You Build | Key Visuals |
|---|--------|-----------|---------------|-------------|
| **00** | Setup | 1 | Environment check | Benchmark charts, roadmap |
| **01** | Foundations | 6 | ViT encoder, text encoder, fusion, attention, transformer, tokenization | Architecture diagrams, patch visualization, attention heatmaps |
| **02** | Vision-Language | 3 | CLIP, image captioner, VQA model | Similarity matrices, attention heatmaps, generation steps |
| **03** | Training | 5 | Contrastive loss, multi-objective, pipeline, alignment, scaling | InfoNCE step-by-step, temperature analysis, scaling curves |
| **04** | Finetuning | 4 | LoRA layer, QLoRA, adapters, CLIP finetune | LoRA diagram, memory comparison, parameter savings |
| **05** | Advanced | 8 | Mini-LLaVA, audio/video, ONNX, diffusion, eval, reasoning, gen, agents | LLaVA architecture, deployment, diffusion sampling |
| **06** | Datasets | 1 | Data loading, preprocessing, quality filtering | Distribution plots, dataset scale comparison |
| | **Total** | **28** | | **60+ visual plots** |

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
| **DPO (Direct Preference Optimization)** | 03 (README) | Yes |
| **GRPO (Group Relative Policy Optimization)** | 03 (README) | Yes |
| **Weight-Space Merging / Model Souping** | 03 (README) | Yes |
| ONNX Export | 05.03 | Yes |
| Post-training Quantization (INT8) | 05.03 | Yes |
| Knowledge Distillation | 05.03 | Yes |
| torch.compile | 05.03 | Yes |

---

## Core Mathematical Concepts

This course covers the complete mathematical pipeline from raw inputs to deployed models. Every equation below is derived step-by-step in the module READMEs and notebooks.

### Stage 1: Encoding — Mapping Inputs to Vectors

**Vision Transformer (ViT)** — split image into patches, project, add positional encoding, process through transformer:

$$\mathbf{x} \in \mathbb{R}^{H \times W \times C} \;\xrightarrow{\text{patch}}\; \{\mathbf{x}_p^i\}_{i=1}^N \;\xrightarrow{E}\; \mathbf{Z}_0 = [\mathbf{z}_{\text{cls}};\, \mathbf{z}_0^1;\, \ldots;\, \mathbf{z}_0^N] + \mathbf{E}_{\text{pos}}$$

**Self-Attention** — the core operation of every transformer:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q = ZW_Q$, $K = ZW_K$, $V = ZW_V$. Division by $\sqrt{d_k}$ is critical — without it, $\text{Var}(q^\top k) = d_k$ causes softmax saturation. Module 01 proves this from first principles.

**Multi-Head Attention** — each head learns a different attention pattern:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i), \quad d_k = d_{\text{model}} / h$$

### Stage 2: Alignment & Fusion — Making Modalities Talk

**Cosine Similarity** — the metric for comparing embeddings across modalities:

$$\text{sim}(\mathbf{v}, \mathbf{t}) = \frac{\mathbf{v}^\top \mathbf{t}}{\|\mathbf{v}\| \|\mathbf{t}\|} = \cos\theta \in [-1, 1]$$

**InfoNCE Loss (CLIP)** — contrastive alignment that lower-bounds mutual information:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} + \log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}} \right]$$

$$I(V; T) \;\geq\; \log N - \mathcal{L}_{\text{InfoNCE}}$$

**Cross-Attention** — one modality queries another for selective fusion:

$$\text{CrossAttn}(Z_{\text{txt}}, Z_{\text{img}}) = \text{softmax}\!\left(\frac{(Z_{\text{txt}} W_Q)(Z_{\text{img}} W_K)^\top}{\sqrt{d_k}}\right)(Z_{\text{img}} W_V)$$

**Autoregressive Generation** — captioning as conditional language modeling:

$$P(\mathbf{y} \mid I) = \prod_{t=1}^{T} P(y_t \mid y_1, \ldots, y_{t-1}, I)$$

### Stage 3: Efficient Finetuning — Training on Limited Hardware

**The Problem** — full finetuning of 7B parameters needs 112 GB VRAM:

$$\text{Memory} = \underbrace{4 \times 7\text{B}}_{\text{weights}} + \underbrace{2 \times 4 \times 7\text{B}}_{\text{Adam } (m, v)} + \underbrace{4 \times 7\text{B}}_{\text{gradients}} = 112 \text{ GB}$$

**LoRA** — low-rank adaptation, exploiting the intrinsic dimensionality of weight updates:

$$h = W_0 x + \frac{\alpha}{r} \underbrace{B A}_{d \times r \;\cdot\; r \times d} x, \quad \text{Savings} = 1 - \frac{2r}{d} \approx 98\%$$

**QLoRA** — quantize the base model to 4-bit NF4, apply LoRA in fp16:

$$h = \text{Dequant}(W_{\text{NF4}}) \cdot x + \frac{\alpha}{r} BAx \quad \Rightarrow \quad 112 \text{ GB} \to 4 \text{ GB}$$

NF4 uses quantile quantization: $q_i = \Phi^{-1}\!\left(\frac{2i+1}{2 \cdot 2^b}\right)$ where $\Phi^{-1}$ is the inverse normal CDF — more levels near zero where most weights live.

### Stage 4: Post-Training Alignment — RLHF, DPO, GRPO

**DPO (Direct Preference Optimization)** — align without a reward model:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

**GRPO (Group Relative Policy Optimization)** — RL with verifiable rewards, no reward model:

$$A_i = \frac{R(x, y_i) - \text{mean}(\lbrace R_j \rbrace)}{\text{std}(\lbrace R_j \rbrace)}, \qquad \mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i} \min\!\left(r_i A_i,\; \text{clip}(r_i, 1\pm\epsilon)\, A_i\right)$$

**Weight-Space Merging** — combine specialized checkpoints without retraining:

$$\theta_{\text{merged}} = \theta^{\text{base}} + \sum_{k=1}^{K} \lambda_k \underbrace{(\theta_k^{\text{fine}} - \theta^{\text{base}})}_{\text{task vector } \tau_k}$$

```
  The Modern 3-Stage Pipeline:

  Stage 1: SFT                Stage 2: RL (DPO/GRPO)       Stage 3: Merge
  ─────────────               ──────────────────            ─────────────
  Next-token pred             Preference / reward           Task arithmetic
  on curated data             optimization                  θ = θ_base + Σ λₖτₖ
       │                           │                              │
       ▼                           ▼                              ▼
  θ_base ──────────────▶ θ_A (task A), θ_B (task B) ──▶ θ_merged (best of all)
```

### Stage 5: Deployment

**ONNX + INT8** — static graph export with post-training quantization:

$$q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right), -128, 127\right), \quad s = \frac{\lvert x \rvert_{\max}}{127}$$

**Knowledge Distillation** — compress a large teacher into a small student:

$$\mathcal{L}_{\text{KD}} = \alpha T^2 \, \text{KL}\!\left(\text{softmax}\!\left(\frac{z_s}{T}\right) \;\Big\|\; \text{softmax}\!\left(\frac{z_t}{T}\right)\right) + (1-\alpha) \mathcal{L}_{\text{CE}}$$

### Summary of Key Equations

| Concept | Equation | Module |
|---------|----------|--------|
| **Patch Embedding** | $\mathbf{z}_i = \mathbf{x}_p^i \cdot \mathbf{E} + \mathbf{e}_{\text{pos}}^i$ | 01 |
| **Self-Attention** | $\text{softmax}(QK^\top / \sqrt{d_k})\, V$ | 01 |
| **LayerNorm** | $\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | 01 |
| **GELU** | $x \cdot \Phi(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$ | 01 |
| **Cosine Similarity** | $\mathbf{v}^\top \mathbf{t} / (\|\mathbf{v}\| \|\mathbf{t}\|)$ | 01, 02 |
| **InfoNCE** | $-\log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$ | 02, 03 |
| **Captioning** | $-\sum_t \log P(y_t \mid y_{1:t-1}, I)$ | 02 |
| **VQA** | $\text{softmax}(W_a \cdot \text{Fuse}(f_I, f_Q))$ | 02 |
| **AdamW** | $\theta_t = \theta_{t-1} - \eta(\hat{m}_t / \sqrt{\hat{v}_t + \epsilon} + \lambda\theta_{t-1})$ | 03 |
| **Cosine LR** | $\frac{\eta_{\max}}{2}(1 + \cos(\pi t / T))$ | 03 |
| **Gradient Accum** | $\bar{g} = \frac{1}{K}\sum_k \nabla\mathcal{L}(\mathcal{B}_k)$ | 03 |
| **LoRA** | $h = W_0 x + \frac{\alpha}{r} BAx$ | 04 |
| **NF4 Quantile** | $q_i = \Phi^{-1}((2i+1) / (2 \cdot 2^b))$ | 04 |
| **IA3** | $K' = \mathbf{l}_K \odot K, \; V' = \mathbf{l}_V \odot V$ | 04 |
| **DPO** | $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$ | 03 |
| **GRPO Advantage** | $A_i = (R_i - \bar{R}) / \text{std}(R)$ | 03 |
| **Task Vector Merge** | $\theta = \theta^{\text{base}} + \sum_k \lambda_k \tau_k$ | 03 |
| **KV Cache** | Memory $= 2 \times L \times T \times d$ | 05 |
| **Mel Scale** | $m = 2595 \log_{10}(1 + f/700)$ | 05 |
| **Distillation** | $\alpha T^2 \text{KL}(\sigma(z_s/T) \| \sigma(z_t/T))$ | 05 |

---

## Hardware Requirements

| Setup | Works? | Notes |
|-------|--------|-------|
| **CPU only (no GPU)** | Yes | All notebooks designed for this |
| **Google Colab Free (T4)** | Yes | Best free option |
| **Laptop GPU (4-8 GB)** | Yes | Faster training |
| **Desktop GPU (12+ GB)** | Yes | Can run pretrained models too |

### Memory Usage

```
  From-scratch models (ALL notebooks):
  ├── RAM:    < 500 MB
  ├── GPU:    Not required
  └── Disk:   Minimal (synthetic data)

  Pretrained model cells (optional, commented out):
  ├── BLIP captioning:    ~1 GB download
  ├── OpenCLIP:           ~600 MB download
  └── QLoRA (7B model):   ~4 GB GPU (with NF4)
```

---

## Key Design Principles

### 1. Visual First
Every concept has a programmatic diagram. No external images — all visuals are generated in code so you can modify and re-run them.

### 2. From Scratch Before Libraries
We build LoRA, CLIP, ViT, attention, and fusion layers from raw PyTorch before showing the PEFT/HuggingFace library versions. You understand what happens inside.

### 3. Mathematically Rigorous
Every equation is derived step-by-step. The module READMEs contain full proofs (e.g., why $\sqrt{d_k}$ scaling prevents softmax saturation, how InfoNCE lower-bounds mutual information, why NF4 outperforms uniform quantization).

### 4. Low Compute by Default
- Models are small (128-256 dim, 2-4 layers)
- Data is synthetic (no downloads)
- Training takes seconds, not hours
- GPU code is always optional

### 5. Progressive Complexity
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
| Slow training | Expected on CPU — models are small so it finishes in seconds |
| LaTeX not rendering | GitHub renders `$...$` math natively; some viewers may not |

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
| [ViT](https://arxiv.org/abs/2010.11929) | 2020 | Vision Transformer — images as token sequences |
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | Contrastive image-text pretraining at scale |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | Low-rank adaptation for parameter efficiency |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 2023 | Bootstrapped vision-language pretraining |
| [LLaVA](https://arxiv.org/abs/2304.08485) | 2023 | Visual instruction tuning for multimodal LLMs |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 2023 | 4-bit quantized LoRA — 7B on a single GPU |
| [ImageBind](https://arxiv.org/abs/2305.05665) | 2023 | 6-modality binding through shared embedding |
| [SigLIP](https://arxiv.org/abs/2303.15343) | 2023 | Sigmoid loss replaces softmax for contrastive learning |

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
git clone https://github.com/<your-username>/Multimodal-Deep-Learning.git
cd Multimodal-Deep-Learning

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

- GitHub: [@Gaurav14cs17](https://github.com/Gaurav14cs17)

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
| **1.4.0** | June 2026 | End-to-end completion: filled 10 scaffold notebooks, added Module 06 (Datasets & Benchmarks), paper references with arXiv links |
| **1.3.0** | June 2026 | Comprehensive audit: 10 new notebooks (attention, transformer, tokenization, alignment, scaling, diffusion, eval, reasoning, generation, agents); full derivations added to all READMEs |
| **1.2.0** | June 2026 | Added RLHF/DPO/GRPO, weight-space merging, LoRA+RL integration across Modules 03–05 |
| **1.1.0** | June 2026 | Deep mathematical overhaul — full derivations, worked examples, proofs in all module READMEs |
| **1.0.0** | June 2026 | Initial release — 17 notebooks, 6 modules, 54 visualizations |

---

**Total: 28 notebooks | 10+ models from scratch | 15+ training techniques | 60+ visual plots**

**Estimated total time: 20-26 hours of focused learning.**

Happy learning!
