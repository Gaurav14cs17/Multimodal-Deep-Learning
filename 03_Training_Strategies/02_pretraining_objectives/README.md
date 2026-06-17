# 02 — Pretraining Objectives for Multimodal Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives/02_pretraining_objectives.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Intermediate–Advanced | **GPU Required:** No

---

## What You'll Learn

Modern VLMs use **multiple objectives simultaneously**. This notebook covers all four and shows how they complement each other:

| Objective | What It Teaches | Granularity | Representation |
|-----------|---------------|------------|----------------|
| **ITC** (Image-Text Contrastive) | Global alignment | Coarse | Shared embedding |
| **ITM** (Image-Text Matching) | Match discrimination | Medium | Cross-modal features |
| **MLM** (Masked Language Modeling) | Visual grounding | Fine | Token-level prediction |
| **Generation** | Fluent captioning | Fine | Autoregressive output |

---

## Multi-Objective Loss

$$
\mathcal{L} = \alpha\,\mathcal{L}_{\text{ITC}} + \beta\,\mathcal{L}_{\text{ITM}} + \gamma\,\mathcal{L}_{\text{MLM}} + \delta\,\mathcal{L}_{\text{Gen}}
$$

Typical weight settings:

| Objective | Weight | Justification |
|-----------|--------|---------------|
| ITC | $\alpha = 1.0$ | Primary alignment signal |
| ITM | $\beta = 1.0$ | Fine-grained discrimination |
| MLM | $\gamma = 1.0$ | Visual grounding |
| Gen | $\delta = 1.0$ | Language fluency |

---

## Objective 1: ITC (Image-Text Contrastive)

The same contrastive loss from CLIP — aligns global image and text representations:

$$
\mathcal{L}_{\text{ITC}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} + \log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}\right]
$$

**What it teaches:** "This image and this text are about the same thing" — **coarse-grained global alignment**.

```
  Image ──► [ViT] ──► [CLS] ──► v ──┐
                                      ├──► S = v⊤t / τ ──► Softmax ──► InfoNCE
  Text  ──► [BERT] ──► [CLS] ──► t ──┘
```

**Limitation:** Only uses [CLS] tokens — ignores fine-grained patch-token correspondence.

---

## Objective 2: ITM (Image-Text Matching)

Binary classification: "Does this image-text pair match?"

$$
\mathcal{L}_{\text{ITM}} = -\left[y \log P_{\text{match}} + (1-y) \log(1 - P_{\text{match}})\right]
$$

where $y \in \{0, 1\}$ and:

$$
P_{\text{match}} = \sigma\!\left(W_{\text{itm}}^\top \cdot \text{CrossEncoder}(f_I, f_T)[0] + b_{\text{itm}}\right)
$$

### Hard Negative Mining for ITM

Not all negatives are equally useful. BLIP uses **contrastive scores from ITC** to mine hard negatives:

```
  For image i, find hard negative text:
  j_hard = argmax_{j ≠ i} S_{ij}     ← most confusing mismatch

  For text j, find hard negative image:
  i_hard = argmax_{i ≠ j} S_{ij}     ← most confusing mismatch
```

**ITM training batch composition:**

| Sample Type | Fraction | Example |
|-------------|----------|---------|
| Positive (matched) | 1/3 | (dog image, "a golden retriever") |
| Hard negative (ITC-mined) | 1/3 | (dog image, "a labrador puppy") |
| Random negative | 1/3 | (dog image, "a red sports car") |

**What it teaches:** Fine-grained discrimination — the model must look beyond global semantics to decide if a specific image matches a specific caption.

---

## Objective 3: MLM (Masked Language Modeling)

Randomly mask 15% of text tokens and predict them using **both** text context and image features:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P(y_t \mid \mathbf{y}_{\setminus \mathcal{M}}, I)
$$

where $\mathcal{M}$ is the set of masked positions.

### BERT-style Masking Protocol

For each selected token (15% of all tokens):

| Action | Probability | Example |
|--------|------------|---------|
| Replace with `[MASK]` | 80% | "a [MASK] on a mat" |
| Replace with random token | 10% | "a **book** on a mat" |
| Keep original | 10% | "a **cat** on a mat" |

### Cross-Modal MLM vs. Text-Only MLM

```
  Text-only MLM (BERT):
  "a [MASK] sitting on a [MASK]" ──► [BERT] ──► "a CAT sitting on a MAT"
  (Must guess from text context alone)

  Cross-Modal MLM (BLIP):
  "a [MASK] sitting on a [MASK]"  ──┐
                                     ├──► [CrossEncoder] ──► "a CAT sitting on a MAT"
  Image of cat on mat             ──┘
  (Can "cheat" by looking at the image!)
```

**What it teaches:** Visual grounding — the model learns which image regions correspond to which words. To predict `[MASK]` = "cat", it must attend to the cat region in the image.

### MLM Loss — Step-by-Step Numerical Example

**Setup:** Vocabulary size $V=5$, masked token at position $t=3$ (ground truth: "cat" = index 2).

**Step 1:** Model outputs logits $\mathbf{z} = [0.5, 1.0, 3.2, 0.8, 0.3]$ for positions $\{$dog, bird, cat, mat, the$\}$.

**Step 2:** Softmax probabilities:

$$
P(\text{cat}) = \frac{e^{3.2}}{e^{0.5} + e^{1.0} + e^{3.2} + e^{0.8} + e^{0.3}} = \frac{24.53}{1.65 + 2.72 + 24.53 + 2.23 + 1.35} = \frac{24.53}{32.48} \approx 0.755
$$

**Step 3:** Cross-entropy loss for this masked token:

$$
\mathcal{L}_{\text{MLM}} = -\log(0.755) \approx 0.281
$$

**Step 4:** With image grounding — if cross-attention correctly attends to cat region, logits improve to $\mathbf{z}' = [0.2, 0.5, 4.5, 0.3, 0.1]$:

$$
P(\text{cat}) \approx 0.945, \quad \mathcal{L}_{\text{MLM}} = -\log(0.945) \approx 0.057
$$

Cross-modal MLM reduces loss by 80% when visual grounding is correct.

---

## Mathematical Proofs

### Proof: MLM Loss — From Masked Positions to Cross-Entropy

**Step 1 — Mask set:** $\mathcal{M} \subset \{1, \ldots, T\}$, masked tokens replaced per BERT protocol.

**Step 2 — Conditional distribution:**

$$
P(y_t \mid \mathbf{y}_{\setminus \mathcal{M}}, I) = \text{softmax}(W_{\text{out}} \mathbf{h}_t)_ {y_t}
$$

where $\mathbf{h}_t$ comes from cross-modal encoder with image $I$.

**Step 3 — Cross-entropy over masked positions only:**

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P(y_t^* \mid \mathbf{y}_{\setminus \mathcal{M}}, I)
$$

**Step 4 — Per-token expansion:**

$$
\log P(y_t^* \mid \cdot) = z_{y_t^*} - \log\sum_{v=1}^{V} e^{z_v}
$$

**Why:** Only masked positions contribute — unmasked tokens provide context via bidirectional attention. **∎**

#### Numerical Example

From above: $z_{\text{cat}} = 3.2$, $\sum_v e^{z_v} = 32.48$, $P(\text{cat}) = 0.755$, $\mathcal{L} = -\log(0.755) = 0.281$. With visual grounding: $P(\text{cat}) = 0.945$, $\mathcal{L} = 0.057$.

---

### Proof: Multi-Objective Training — Why ITC + ITM + MLM Helps

**Step 1 — Joint loss:**

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{ITC}} + \beta \mathcal{L}_{\text{ITM}} + \gamma \mathcal{L}_{\text{MLM}}
$$

**Step 2 — Gradient decomposition:**

$$
\nabla_\theta \mathcal{L} = \alpha \nabla \mathcal{L}_{\text{ITC}} + \beta \nabla \mathcal{L}_{\text{ITM}} + \gamma \nabla \mathcal{L}_{\text{MLM}}
$$

**Step 3 — Complementary gradients:**

| Objective | Gradient acts on | What it teaches |
|-----------|-----------------|-----------------|
| ITC | Projection heads, [CLS] | Global alignment |
| ITM | Cross-encoder fusion | Match vs non-match discrimination |
| MLM | Token-level cross-attn | Visual grounding per word |

**Step 4 — Why multi-objective helps:** ITC alone cannot distinguish hard negatives (similar captions); ITM adds binary discrimination; MLM forces patch-level alignment — gradients reach different layers and prevent collapse to coarse features. **∎**

#### Numerical Example

BLIP ablation: ITC only R@1 = 78.4%; +ITM → 82.1%; +MLM → 83.5%. Each added objective reduces retrieval error by ~2–4% — gradients from MLM improve cross-attention weights used indirectly by ITC.

---

## Objective 4: Generation (Autoregressive Captioning)

Generate captions token by token, conditioned on the image:

$$
\mathcal{L}_{\text{Gen}} = -\sum_{t=1}^{T} \log P(y_t \mid y_1, \ldots, y_{t-1}, I)
$$

```
  Image ──► [ViT] ──► visual features
                           │
                           ▼ (cross-attention)
  <bos> ──► [Causal Decoder] ──► "A" ──► "cat" ──► "on" ──► "mat" ──► <eos>
```

**Key architectural difference:** The decoder uses **causal self-attention** (left-to-right), unlike the bidirectional encoder used for ITC/ITM/MLM.

**What it teaches:** Language fluency — the model learns to produce grammatically correct, descriptive sentences about images.

---

## BLIP Architecture — All Four Objectives

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                        BLIP Architecture                        │
  │                                                                 │
  │  Image ──► [ViT] ──► v ──┬──► ITC head (contrastive loss)      │
  │                           │    Uses: unimodal v and t only      │
  │                           │                                     │
  │                           ├──► [Cross-Modal Encoder]            │
  │                           │    │                                │
  │  Text  ──► [BERT] ──► t ──┘    ├──► ITM head (binary: match?)  │
  │                                │    Uses: fused cross-modal     │
  │                                │                                │
  │                                ├──► MLM head (predict [MASK])   │
  │                                │    Uses: fused + masked input  │
  │                                │                                │
  │                                └──► [Causal Decoder]            │
  │                                     └──► Gen head (caption)     │
  │                                          Uses: causal + image   │
  └─────────────────────────────────────────────────────────────────┘
```

### Weight Sharing in BLIP

| Component | Shared Between | Purpose |
|-----------|---------------|---------|
| ViT image encoder | All 4 objectives | Single visual representation |
| Text embedding layer | ITC, ITM, MLM, Gen | Shared vocabulary |
| Cross-attention layers | ITM, MLM, Gen | Cross-modal interaction |
| Self-attention layers | NOT shared between encoder/decoder | Different attention patterns |

---

## How Objectives Complement Each Other

```
  ITC:  "This image is about animals"         ← Coarse alignment
         │
         ▼
  ITM:  "This specific cat image matches      ← Fine discrimination
         this specific caption"
         │
         ▼
  MLM:  "The word [MASK] in position 3 is     ← Token-level grounding
         'cat' because that's what's in
         the image"
         │
         ▼
  Gen:  "A fluffy orange cat sitting           ← Full generation
         comfortably on a red mat"
```

### Ablation Study (from BLIP paper)

| Objectives Used | Image-Text Retrieval (R@1) | Captioning (CIDEr) |
|----------------|--------------------------|---------------------|
| ITC only | 78.4 | — |
| ITC + ITM | 82.1 | — |
| ITC + ITM + MLM | 83.5 | — |
| ITC + ITM + Gen | 82.8 | 133.2 |
| ITC + ITM + MLM + Gen | **84.2** | **136.7** |

---

## Budget → Objectives → Model Mapping

| Budget | Objectives | Architecture | Example Models |
|--------|-----------|-------------|---------------|
| Low | ITC only | Dual encoder (no cross-attn) | CLIP, ALIGN |
| Medium | ITC + ITM + MLM | Encoder + cross-modal encoder | BLIP, ALBEF |
| High | ITC + ITM + MLM + Gen | Full encoder-decoder | BLIP-2, CoCa |
| Very High | All + Routing + MoE | Mixture of experts | Gemini, GPT-4V |

---

## What You'll Build

- All 4 objectives implemented from scratch
- Multi-objective training loop with configurable weights
- Ablation experiment: measure the effect of adding each objective
- Visualization of MLM attention maps (which image regions help predict masked words)

---

## Prerequisites

- Module 03 Notebook 01 (contrastive learning deep dive)
- Understanding of cross-entropy, binary cross-entropy
- Familiarity with BERT masking and autoregressive generation

---

## 🔬 Worked Examples in the Notebook

### Multi-Objective Training Loop
- Train ITC + ITM + MLM simultaneously with weighted objectives
- 60 epochs with per-objective loss tracking
- Ablation study: which objectives help most?
- Visualize training curves and ablation bar chart

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/02_pretraining_objectives/02_pretraining_objectives.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| Pretraining Landscape | Multiple papers | 2021-23 | Which model uses which objectives (ITC/ITM/MLM/LM/MIM) |

### Additional Papers Covered

- **CoCa** (Yu et al., 2022) — Contrastive + captioning dual decoder, split unimodal/multimodal
- **BEiT-3** (Wang et al., 2022) — Unified masked modeling across vision, language, and multimodal
- **ALBEF** (Li et al., 2021) — Align before fuse, momentum distillation

---

## Next Step

**[03_training_pipeline](../03_training_pipeline/)** — Full production training pipeline with all best practices
