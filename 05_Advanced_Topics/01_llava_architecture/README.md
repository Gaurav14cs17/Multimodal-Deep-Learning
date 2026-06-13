# 01 — LLaVA Architecture: Connecting Vision to LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/01_llava_architecture/01_llava_architecture.ipynb)

> **Time:** ~50 minutes | **Difficulty:** Advanced | **GPU Required:** No (mini version)

---

## What You'll Learn

**LLaVA (Large Language and Vision Assistant)** is the blueprint for modern multimodal LLMs like GPT-4V and Gemini.

- LLaVA architecture — step-by-step diagram with tensor shapes
- The projection layer: how 576 image patches become "language tokens"
- Causal attention with visual prefix tokens
- KV cache for efficient generation
- Two-stage training (alignment → instruction tuning)
- Build a Mini-LLaVA from scratch

---

## Architecture — Full Tensor Flow

```
  ┌────────────────────────────────────────────────────────────────┐
  │                     LLaVA Architecture                         │
  │                                                                │
  │  Image (224×224×3)                                             │
  │       │                                                        │
  │       ▼                                                        │
  │  ┌──────────────────┐                                          │
  │  │ CLIP ViT-L/14     │  576 patches + [CLS]                   │
  │  │  (frozen)         │  Output: Z_clip ∈ ℝ^(576 × 1024)       │
  │  │  304M params      │  (discard [CLS], keep patch features)  │
  │  └────────┬─────────┘                                          │
  │           │                                                    │
  │           ▼                                                    │
  │  ┌──────────────────┐                                          │
  │  │ MLP Projector     │  2-layer MLP with GELU                 │
  │  │  (trainable)      │  1024 → 4096 → 4096                    │
  │  │  ~8M params       │  Output: H_v ∈ ℝ^(576 × 4096)         │
  │  └────────┬─────────┘                                          │
  │           │                                                    │
  │           ▼                                                    │
  │  ┌────────────────────────────────────────────────────┐       │
  │  │  Combined Token Sequence                            │       │
  │  │                                                     │       │
  │  │  [v₁, v₂, ..., v₅₇₆,  <sys>,  t₁, t₂, ..., tₙ]  │       │
  │  │  └─── visual tokens ───┘ └───── text tokens ──────┘│       │
  │  │       576 tokens              n tokens              │       │
  │  │                                                     │       │
  │  │  Total sequence: 576 + n tokens, each ∈ ℝ^4096     │       │
  │  └────────────────────┬───────────────────────────────┘       │
  │                       │                                        │
  │                       ▼                                        │
  │  ┌──────────────────────────────────────────┐                  │
  │  │  LLM (Vicuna 7B / LLaMA)                 │                  │
  │  │  32 Transformer layers                    │                  │
  │  │  d_model = 4096, h = 32                   │                  │
  │  │  Causal attention (left-to-right)         │                  │
  │  │  + LoRA adapters (r=128)                  │                  │
  │  └──────────────────┬───────────────────────┘                  │
  │                     │                                          │
  │                     ▼                                          │
  │              Generated response                                │
  │         "The image shows a cat sitting on a red mat."          │
  └────────────────────────────────────────────────────────────────┘
```

---

## Key Equations

### 1. Visual Projection — MLP Projector

$$
\mathbf{h}_v = W_2 \cdot \text{GELU}(W_1 \cdot \mathbf{z}_{\text{CLIP}} + b_1) + b_2
$$

where:
- $W_1 \in \mathbb{R}^{1024 \times 4096}$ — maps CLIP dimension to LLM dimension
- $W_2 \in \mathbb{R}^{4096 \times 4096}$ — second layer
- GELU non-linearity between layers

**Why MLP, not linear?** LLaVA-1.0 used a simple linear projection; LLaVA-1.5 found that a 2-layer MLP improved performance by ~2% on benchmarks.

### 2. Combined Sequence — How Vision Becomes Language

The visual tokens are treated as **regular input tokens** by the LLM:

$$
X = [\underbrace{\mathbf{h}_{v_1}, \ldots, \mathbf{h}_{v_{576}}}_{\text{visual tokens}}, \underbrace{\mathbf{e}_{\text{sys}}, \mathbf{e}_{t_1}, \ldots, \mathbf{e}_{t_n}}_{\text{text tokens}}]
$$

where $\mathbf{e}_{t_i}$ are the text token embeddings from the LLM's embedding layer.

### 3. Causal Attention with Visual Prefix

In the LLM, each token attends to all **previous** tokens (causal mask):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{causal}}\right) V
$$

**Critical insight:** Text tokens can attend to ALL visual tokens (they appear first in the sequence), enabling rich visual reasoning:

```
  Attention mask (simplified, 4 visual + 3 text tokens):

         v₁  v₂  v₃  v₄  t₁  t₂  t₃
  v₁  [  ✓   ✗   ✗   ✗   ✗   ✗   ✗  ]
  v₂  [  ✓   ✓   ✗   ✗   ✗   ✗   ✗  ]
  v₃  [  ✓   ✓   ✓   ✗   ✗   ✗   ✗  ]
  v₄  [  ✓   ✓   ✓   ✓   ✗   ✗   ✗  ]
  t₁  [  ✓   ✓   ✓   ✓   ✓   ✗   ✗  ]  ← sees ALL visual tokens
  t₂  [  ✓   ✓   ✓   ✓   ✓   ✓   ✗  ]
  t₃  [  ✓   ✓   ✓   ✓   ✓   ✓   ✓  ]
```

### 4. KV Cache — Efficient Autoregressive Generation

During generation, previously computed Key and Value vectors are **cached**:

$$
K_{\text{cache}} = [K_1, K_2, \ldots, K_{t-1}], \quad V_{\text{cache}} = [V_1, V_2, \ldots, V_{t-1}]
$$

At step $t$, only compute $Q_t, K_t, V_t$ for the **new token**:

$$
\text{Attn}_t = \text{softmax}\!\left(\frac{Q_t [K_{\text{cache}}; K_t]^\top}{\sqrt{d_k}}\right) [V_{\text{cache}}; V_t]
$$

**KV cache memory for LLaVA:**

$$
\text{KV Memory} = 2 \times L \times T \times d \times \text{bytes}
$$

For LLaVA-7B ($L = 32$, $d = 4096$, $T = 576 + 256 = 832$ tokens, fp16):

$$
\text{KV Memory} = 2 \times 32 \times 832 \times 4096 \times 2 \text{ bytes} = 436 \text{ MB}
$$

Note: The 576 visual tokens are a significant portion of the KV cache! This is why efficient visual token compression is an active research area.

---

## Two-Stage Training

### Stage 1: Visual-Language Alignment (Pretraining)

**Freeze:** ViT encoder + LLM
**Train:** MLP projector only

$$
\mathcal{L}_{\text{align}} = -\sum_{t \in \text{response}} \log P(y_t \mid y_1, \ldots, y_{t-1}, \mathbf{H}_v)
$$

Data: 558K image-caption pairs from CC3M (filtered).

**Purpose:** Teach the projector to map CLIP features into the LLM's "language" so visual tokens are interpretable.

### Stage 2: Instruction Tuning (Finetuning)

**Freeze:** ViT encoder
**Train:** MLP projector + LLM (with LoRA)

$$
\mathcal{L}_{\text{instruct}} = -\sum_{t \in \text{response}} \log P(y_t \mid y_1, \ldots, y_{t-1}, \mathbf{H}_v, \text{instruction})
$$

Data: 150K GPT-4-generated multimodal instruction-following data.

```
  Stage 1: Alignment         Stage 2: Instruction Tuning
  ┌─────────────────┐        ┌─────────────────┐
  │ ViT   [frozen]  │        │ ViT   [frozen]  │
  │ MLP   [TRAIN]   │        │ MLP   [TRAIN]   │
  │ LLM   [frozen]  │        │ LLM   [LoRA]    │
  │                  │        │                  │
  │ Data: captions   │        │ Data: instruct   │
  │ 558K pairs       │        │ 150K examples    │
  │ ~1 epoch         │        │ ~3 epochs        │
  └─────────────────┘        └─────────────────┘
```

---

## Parameter Count Breakdown

| Component | Parameters | Trainable (Stage 1) | Trainable (Stage 2) |
|-----------|-----------|-------------------|-------------------|
| CLIP ViT-L/14 | 304M | Frozen | Frozen |
| MLP Projector | 8M | 8M | 8M |
| LLM (Vicuna 7B) | 6.7B | Frozen | LoRA: ~20M |
| **Total trainable** | | **8M (0.1%)** | **28M (0.4%)** |

---

## LLaVA Variants Comparison

| Model | Vision Encoder | LLM | Projector | Key Difference |
|-------|---------------|-----|-----------|---------------|
| LLaVA-1.0 | CLIP ViT-L/14 | Vicuna 7B | Linear | First open-source VLM |
| LLaVA-1.5 | CLIP ViT-L/14@336 | Vicuna 7B/13B | 2-layer MLP | Better projector + higher res |
| LLaVA-1.6 (NeXT) | CLIP ViT-L/14 | Vicuna/Mistral | MLP | Dynamic resolution + multi-image |
| BLIP-2 | ViT-G | FlanT5/OPT | Q-Former (32 queries) | Fewer visual tokens (32 vs 576) |
| Qwen-VL | ViT-G | Qwen 7B | Cross-attention | Position-aware visual features |

---

## What You'll Build

- Mini-LLaVA from scratch (small ViT + small LLM + MLP projector)
- Projector training (Stage 1)
- Instruction tuning with LoRA (Stage 2)
- KV cache implementation for efficient generation
- Visual token attention analysis

---

## Prerequisites

- Module 02 (CLIP, captioning)
- Module 04 (LoRA, adapter methods)
- Understanding of autoregressive LLMs
- Familiarity with causal masking

---

## Next Step

**[02_multimodal_beyond_vision](../02_multimodal_beyond_vision/)** — Audio, video, and 6-modality models
