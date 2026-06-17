# 03 — Visual Question Answering (VQA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/03_visual_question_answering/03_visual_question_answering.ipynb)

> **Time:** ~35 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

VQA is where vision meets language reasoning — **"look at this image, answer this question."**

- VQA as classification: Image + Question → one of $K$ answers
- Cross-attention fusion between image and question features
- Attention visualization: which image regions answer which questions
- Comparison of VQA approaches (classifier vs generative vs retrieval)
- Building a cross-attention VQA model from scratch

---

## Architecture — Cross-Attention VQA

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    VQA Architecture                          │
  │                                                              │
  │  IMAGE:                          QUESTION:                   │
  │  ┌────────────┐                  ┌────────────┐              │
  │  │ Image      │                  │ "What      │              │
  │  │ (224×224)  │                  │  animal    │              │
  │  └─────┬──────┘                  │  is this?" │              │
  │        │                         └─────┬──────┘              │
  │        ▼                               ▼                     │
  │  ┌────────────┐                  ┌────────────┐              │
  │  │ ViT        │                  │ BERT       │              │
  │  │ Encoder    │                  │ Encoder    │              │
  │  └─────┬──────┘                  └─────┬──────┘              │
  │        │                               │                     │
  │        │ f_I ∈ ℝ^(N×D)                │ f_Q ∈ ℝ^(M×D)      │
  │        │ (196 patch features)          │ (token features)    │
  │        │                               │                     │
  │        └───────────┬───────────────────┘                     │
  │                    │                                          │
  │                    ▼                                          │
  │  ┌──────────────────────────────────────────┐               │
  │  │        Cross-Attention Fusion             │               │
  │  │  Q = f_Q · W_Q                           │               │
  │  │  K = f_I · W_K,   V = f_I · W_V          │               │
  │  │  Attended = softmax(QK⊤/√d) · V          │               │
  │  └──────────────────┬───────────────────────┘               │
  │                     │                                        │
  │                     ▼                                        │
  │  ┌──────────────────────────────────────────┐               │
  │  │  Pooling → Classifier → softmax → answer │               │
  │  └──────────────────────────────────────────┘               │
  │                                                              │
  │  Output: "cat" (from K = 3129 answer vocabulary)             │
  └─────────────────────────────────────────────────────────────┘
```

---

## Key Equations

### 1. Cross-Attention Fusion

The question features **query** the image features:

$$
\text{Attended}(f_Q, f_I) = \text{softmax}\!\left(\frac{(f_Q W_Q)(f_I W_K)^\top}{\sqrt{d_k}}\right)(f_I W_V)
$$

**Interpretation:** Each question token learns *which image regions* are relevant. For "What animal?", the attention concentrates on the animal in the image.

### 2. Attention Weight Visualization

The attention matrix $\alpha \in \mathbb{R}^{M \times N}$ ($M$ question tokens $\times$ $N$ image patches):

$$
\alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{l=1}^{N} \exp(q_i^\top k_l / \sqrt{d_k})}
$$

```
  Attention map for question: "What color is the car?"

  "What"   → [0.02, 0.02, 0.03, ..., 0.01]  (spread evenly)
  "color"  → [0.01, 0.41, 0.38, ..., 0.01]  (attends to car body patches)
  "is"     → [0.02, 0.02, 0.02, ..., 0.02]  (spread evenly)
  "the"    → [0.01, 0.01, 0.01, ..., 0.01]  (spread evenly)
  "car"    → [0.01, 0.25, 0.32, ..., 0.22]  (attends to car patches)
```

### 3. Pooling Strategies

After cross-attention, we pool the fused features into a single vector:

| Method | Formula | When |
|--------|---------|------|
| [CLS] pooling | $\mathbf{h} = \text{Attended}[0]$ | BERT-style, standard |
| Mean pooling | $\mathbf{h} = \frac{1}{M}\sum_{i=1}^{M} \text{Attended}_i$ | Better for long questions |
| Attention pooling | $\mathbf{h} = \sum_i \alpha_i \cdot \text{Attended}_i$ | Learned weighting |

### 4. Classification Head

$$
\hat{a} = \arg\max_a \; P(a \mid I, Q) = \arg\max_a \;\text{softmax}(W_a \cdot \mathbf{h} + b_a)
$$

where $W_a \in \mathbb{R}^{K \times D}$, $K$ is the answer vocabulary size.

**Training loss (multi-label soft cross-entropy):**

VQA uses soft labels because 10 annotators may give different answers:

$$
\mathcal{L} = -\sum_{a=1}^{K} \hat{s}_a \log P(a \mid I, Q)
$$

where the soft score $\hat{s}_a = \min(1, \text{count}_a / 3)$ — an answer gets full credit if $\geq 3$ annotators agree.

### VQA Loss — Step-by-Step Numerical Example

**Setup:** $K=4$ answer classes `{cat, dog, red, blue}`, 3 annotators gave: `[cat, cat, dog]`.

**Step 1:** Compute soft scores:

$$
\hat{s}_{\text{cat}} = \min(1, 2/3) = 0.667, \quad \hat{s}_{\text{dog}} = \min(1, 1/3) = 0.333, \quad \hat{s}_{\text{red}} = 0, \quad \hat{s}_{\text{blue}} = 0
$$

**Step 2:** Model logits $\mathbf{z} = [2.1, 0.5, -1.0, -0.5]$ → softmax:

$$
P = \text{softmax}(\mathbf{z}) \approx [0.821, 0.123, 0.034, 0.022]
$$

**Step 3:** Soft cross-entropy loss:

$$
\mathcal{L} = -(0.667 \log 0.821 + 0.333 \log 0.123) = -(0.667 \times (-0.197) + 0.333 \times (-2.095)) = 0.131 + 0.698 = 0.829
$$

If model predicted dog with high confidence instead, loss would increase because $\hat{s}_{\text{dog}} = 0.333$ still requires some probability mass on "dog".

---

## Mathematical Proofs

### Proof: Soft Cross-Entropy for VQA — KL Divergence Connection

**Theorem:** VQA soft-target loss equals cross-entropy with a smoothed label distribution; minimizing it is equivalent to minimizing KL divergence to annotator consensus.

**Step 1 — Annotator distribution:**

For answer $a$, soft score $\hat{s}_a = \min(1, \text{count}_a / 3)$ where count is from 10 annotators (VQA v2 protocol).

**Step 2 — Model prediction:**

$$
P(a \mid I, Q) = \text{softmax}(W_a \mathbf{h} + b_a)_a
$$

**Step 3 — Soft cross-entropy:**

$$
\mathcal{L} = -\sum_{a=1}^{K} \hat{s}_a \log P(a \mid I, Q)
$$

**Step 4 — KL connection:** Let $\hat{p}_a = \hat{s}_a / \sum_{a'} \hat{s}_{a'}$ (normalized soft labels). Then:

$$
\mathcal{L} = H(\hat{p}, P) = H(\hat{p}) + \text{KL}(\hat{p} \,\|\, P)
$$

Since $\hat{p}$ is fixed, minimizing $\mathcal{L}$ minimizes $\text{KL}(\hat{p} \,\|\, P)$ — matching the annotator distribution, not just the majority vote. **∎**

#### Numerical Example

From above: $\hat{s} = [0.667, 0.333, 0, 0]$, $P = [0.821, 0.123, 0.034, 0.022]$:

$$
\mathcal{L} = -(0.667 \log 0.821 + 0.333 \log 0.123) = 0.131 + 0.698 = 0.829
$$

If model put all mass on "dog" ($P_{\text{dog}}=1$): $\mathcal{L} = -0.333 \log 1 = 0$ for dog term but $-0.667 \log 0 = \infty$ — soft loss penalizes ignoring minority "cat" votes.

---

### Proof: Cross-Attention for VQA — Q=Question, K,V=Image

**Step 1 — Encode modalities separately:**

$$
f_I = \text{ViT}(I) \in \mathbb{R}^{N \times D}, \quad f_Q = \text{BERT}(Q) \in \mathbb{R}^{M \times D}
$$

**Step 2 — Question queries image:**

$$
Q = f_Q W_Q, \quad K = f_I W_K, \quad V = f_I W_V
$$

**Why:** The question asks "what animal?" — query vectors encode the semantic need; image patches supply keys/values.

**Step 3 — Attention weights reveal grounding:**

$$
\alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{l=1}^{N} \exp(q_i^\top k_l / \sqrt{d_k})}
$$

For token "animal", $\alpha_{i,:}$ peaks at patches containing the animal.

**Step 4 — Pool and classify:**

$$
\mathbf{h} = \text{Pool}(\text{CrossAttn}(f_Q, f_I)), \quad \hat{a} = \arg\max_a \,\text{softmax}(W_a \mathbf{h})_a
$$

**∎**

#### Numerical Example

Question token "color" attends to patches $\alpha = [0.01, 0.41, 0.38, 0.20]$ — 79% mass on car-body patches (indices 1–2), enabling answer "red."

---

## VQA Approaches — Detailed Comparison

### Approach 1: Classification (Traditional VQA)

$$
P(a \mid I, Q) = \text{softmax}(W_a \cdot \text{Fuse}(f_I, f_Q))
$$

| Pros | Cons |
|------|------|
| Fast inference | Fixed answer vocabulary |
| Simple training | Can't handle open-ended questions |
| Well-studied | Needs pre-defined answer set |

### Approach 2: Generative (Modern VLMs)

$$
P(\text{answer} \mid I, Q) = \prod_{t=1}^{T} P(a_t \mid a_1, \ldots, a_{t-1}, I, Q)
$$

| Pros | Cons |
|------|------|
| Open-ended answers | Slower (autoregressive) |
| Can explain reasoning | Harder to train |
| Flexible output format | Evaluation is harder |

### Approach 3: Retrieval-Based

$$
\hat{a} = \arg\max_{a \in \mathcal{C}} \text{sim}(\text{Fuse}(f_I, f_Q), \text{Embed}(a))
$$

| Pros | Cons |
|------|------|
| Scalable | Needs candidate pool |
| Can use pre-computed embeddings | Limited to candidates |

---

## Question Type Analysis

| Question Type | Example | Required Reasoning | % of VQA v2 |
|---------------|---------|-------------------|-------------|
| Yes/No | "Is there a dog?" | Object detection | ~38% |
| Number | "How many cats?" | Counting | ~12% |
| Color | "What color is the car?" | Attribute recognition | ~8% |
| What | "What is the person doing?" | Activity recognition | ~30% |
| Where | "Where is the book?" | Spatial reasoning | ~7% |
| Other | "What brand is this?" | OCR / Knowledge | ~5% |

---

## What You'll Build

- Cross-attention VQA model from scratch
- Training loop on synthetic image-question pairs
- Attention maps showing which regions the model looks at per question
- Comparison of pooling strategies
- Accuracy analysis by question type

---

## Prerequisites

- Notebook 01 (CLIP architecture, contrastive learning)
- Notebook 02 (encoder-decoder, cross-attention)
- Understanding of classification with softmax

---

## 🔬 Worked Examples in the Notebook

### Example 1: VQA Training Loop
Train a VQA classifier on 500 synthetic samples:
- 40 epochs with loss and accuracy tracking
- Per-answer-type accuracy breakdown
- Achieves ~100% on synthetic data with signal injection

### Example 2: Attention Analysis
Visualize cross-attention maps for 5 test samples:
- Image + attention heatmap side by side
- Numerical breakdown: top-3 attended patches per question word
- Interpret what the model "looks at" to answer

### Example 3: Real-World VQA Applications
Applications table: Medical VQA, Document VQA, Chart QA, Remote Sensing, Retail, Accessibility — with model recommendations for each.

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/03_visual_question_answering/03_visual_question_answering.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| BLIP-2 Framework (`../../assets/paper_figures/blip2_framework.png`) | Li et al. — [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) | 2023 | Q-Former bridges frozen ViT + frozen LLM |
| VQA Architecture Evolution (3-panel) | Antol et al. / Anderson et al. / Liu et al. | 2015-23 | Simple fusion → Bottom-up attention → LLM-based VQA |

### Additional Papers Covered

- **InstructBLIP** (Dai et al., 2023) — Instruction-aware Q-Former
- **LLaVA** (Liu et al., 2023) — Visual instruction tuning with MLP projector

---

## Next Step

**[01_contrastive_learning](../../03_Training_Strategies/01_contrastive_learning/)** — Deep dive into the #1 training strategy
