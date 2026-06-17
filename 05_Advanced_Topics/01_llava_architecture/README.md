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

## Mathematical Proofs

### Proof: MLP Projector Dimension Analysis — Why Simple Projection Works

**Step 1 — Dimension mismatch:** CLIP ViT-L outputs $d_{\text{clip}} = 1024$; LLM expects $d_{\text{llm}} = 4096$.

**Step 2 — Linear projector:** $H_v = W_1 Z_{\text{clip}}$ maps $1024 \to 4096$. Parameter count: $1024 \times 4096 \approx 4.2$M.

**Step 3 — 2-layer MLP (LLaVA-1.5):** $H_v = W_2 \cdot \text{GELU}(W_1 Z + b_1) + b_2$ adds nonlinearity — allows nonlinear alignment between visual and language manifolds.

**Step 4 — Why it works:** CLIP features are already semantically structured; projector only needs to learn a coordinate transform + mild nonlinearity, not full cross-modal reasoning (LLM handles that). Empirically, MLP beats linear by ~2% on benchmarks. **∎**

#### Numerical Example

576 patches × 4096 dim = 2.36M floats per image in LLM space. MLP params: $1024 \times 4096 + 4096 \times 4096 \approx 21$M (both layers) — still <0.3% of 7B LLM.

---

### Proof: KV Cache Memory — Full Derivation

**Step 1 — Per-layer cache:** Store keys and values for all prior tokens:

$$
K_{\text{cache}}^{(l)} \in \mathbb{R}^{T \times d}, \quad V_{\text{cache}}^{(l)} \in \mathbb{R}^{T \times d}
$$

**Step 2 — Total across $L$ layers:**

$$
\text{Memory}_{\text{KV}} = 2 \times L \times T \times d \times \text{bytes\_per\_elem}
$$

Factor 2 for both K and V.

**Step 3 — LLaVA sequence length:** $T = N_{\text{visual}} + N_{\text{text}} = 576 + 256 = 832$.

**Step 4 — Numerical evaluation ($L=32$, $d=4096$, fp16):

$$
\text{Memory} = 2 \times 32 \times 832 \times 4096 \times 2 = 436{,}207{,}616 \text{ bytes} \approx 436 \text{ MB}
$$

**Step 5 — Visual token dominance:** $576/832 \approx 69\%$ of KV cache is visual — motivates token compression (Q-Former: 576→32). **∎**

#### Numerical Example

Reduce visual tokens 576→32: new $T = 32 + 256 = 288$. KV memory $= 2 \times 32 \times 288 \times 4096 \times 2 \approx 151$ MB — **65% savings**.

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

## 🔬 Worked Examples in the Notebook

### LLaVA Inference — Complete Tensor Shape Trace
- Trace every tensor through LLaVA-7B: image → ViT → projector → LLM
- 576 visual tokens + 15 text tokens = 591 total (97% are visual!)
- Memory analysis: KV cache calculation for 591 tokens = ~0.6 GB
- Parameter distribution: 304M (ViT, frozen) + 8M (projector) + 6.7B (LLM, LoRA)

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/01_llava_architecture/01_llava_architecture.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| LLaVA Architecture (`../../assets/paper_figures/llava_arch.png`) | Liu et al. — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) | 2023 | CLIP ViT → MLP projector → Vicuna LLM |
| LLaVA Two-Stage Training | Liu et al. — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) | 2023 | Stage 1: alignment (only MLP), Stage 2: instruction tuning (MLP + LLM) |

### Key Papers & References

#### Vision-Language Model Papers

1. **LLaVA: Visual Instruction Tuning** — Liu et al. (2023) — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) — Simple MLP projector + 2-stage training
2. **LLaVA-1.5: Improved Baselines with Visual Instruction Tuning** — Liu et al. (2023) — [arXiv:2310.03744](https://arxiv.org/abs/2310.03744) — Higher resolution, MLP projector improvement
3. **Qwen-VL: A Versatile Vision-Language Model** — Bai et al. (2023) — [arXiv:2308.12966](https://arxiv.org/abs/2308.12966) — Dynamic resolution, multi-image understanding
4. **InternVL: Scaling up Vision Foundation Models** — Chen et al. (2023) — [arXiv:2312.14238](https://arxiv.org/abs/2312.14238) — Scaling open-source MLLMs to GPT-4V level
5. **BLIP-2: Bootstrapping Language-Image Pre-training** — Li et al. (2023) — [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) — Q-Former bridges frozen ViT + frozen LLM
6. **MiniCPM-V: A GPT-4V Level MLLM on Your Phone** — Yao et al. (2024) — [arXiv:2408.01800](https://arxiv.org/abs/2408.01800) — Mobile deployment, efficient architecture
7. **DeepSeek-VL2: Mixture-of-Experts Vision-Language Models** — Lu et al. (2024) — [arXiv:2412.10302](https://arxiv.org/abs/2412.10302) — MoE for VLMs, efficient scaling
8. **Cambrian-1: Vision-Centric Exploration of Multimodal LLMs** — Tong et al. (2024) — [arXiv:2406.16860](https://arxiv.org/abs/2406.16860) — Systematic study of vision encoder choices

#### Unified Understanding + Generation Papers

9. **Emu3: Next-Token Prediction is All You Need** — Wang et al. (2024) — [arXiv:2409.18869](https://arxiv.org/abs/2409.18869) — Native multimodal with next-token prediction
10. **Show-o: One Single Transformer for Unified Understanding and Generation** — Xie et al. (2024) — [arXiv:2408.12528](https://arxiv.org/abs/2408.12528) — Autoregressive + discrete diffusion in one model

#### Blog Posts & Technical Reports

- 📝 [Lilian Weng — "Large Multimodal Models"](https://lilianweng.github.io/posts/2023-06-23-agent/) — Comprehensive survey of MLLM architectures
- 📝 [Chip Huyen — "Building LLM Applications for Production"](https://huyenchip.com/2023/04/11/llm-engineering.html) — Practical deployment guide
- 📝 [Sebastian Raschka — "Understanding Large Language Models"](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — From theory to practice
- 📝 [Jay Alammar — "The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) — Visual guide to transformers
- 📝 [Jay Alammar — "The Illustrated BERT"](https://jalammar.github.io/illustrated-bert/) — Visual guide to BERT
- 📝 [OpenAI CLIP Blog](https://openai.com/research/clip) — Original CLIP announcement
- 📝 [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft/) — PEFT library guide
- 📝 [HuggingFace TRL Documentation](https://huggingface.co/docs/trl/) — RLHF/DPO training

---

## Next Step

**[02_multimodal_beyond_vision](../02_multimodal_beyond_vision/)** — Audio, video, and 6-modality models
