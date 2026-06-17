# 06 — Multimodal Reasoning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/06_multimodal_reasoning/06_multimodal_reasoning.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

Reasoning across vision and language — the frontier of GPT-4V and Gemini:

- Chain-of-Thought (CoT) with visual inputs
- Compositional and spatial reasoning
- Visual grounding and referential expressions
- Mini visual CoT demo

---

## Chain-of-Thought — Derivation

### Step 1: Standard VQA

$$
P(a \mid I, Q) = \text{softmax}(\text{Classifier}(\text{Fuse}(I, Q)))
$$

Direct answer — no intermediate reasoning.

### Step 2: CoT Decomposition

Introduce reasoning chain $R = (r_1, r_2, \ldots, r_T)$:

$$
P(a \mid I, Q) = \sum_R P(R \mid I, Q) \cdot P(a \mid I, Q, R)
$$

Autoregressive factorization:

$$
P(R \mid I, Q) = \prod_{t=1}^{T} P(r_t \mid I, Q, r_{1:t-1})
$$

### Step 3: Training Objective

$$
\mathcal{L} = -\log P(a^* \mid I, Q, R^*) - \lambda \log P(R^* \mid I, Q)
$$

where $R^*, a^*$ are annotated reasoning steps and answer.

**Numerical example:** Question "How many red objects?"

| Step | Reasoning token $r_t$ | $P(r_t \mid \cdot)$ |
|------|----------------------|---------------------|
| 1 | "I see 3 objects" | 0.85 |
| 2 | "2 are red, 1 is blue" | 0.78 |
| 3 | "Answer: 2" | 0.92 |

Joint probability: $0.85 \times 0.78 \times 0.92 \approx 0.61$

---

## Compositional Reasoning

Query requires combining visual attributes:

$$
P(\text{"red square left of blue circle"} \mid I) = P(c_1 \mid I) \cdot P(c_2 \mid I) \cdot P(r \mid c_1, c_2, I)
$$

where $c_1, c_2$ are detected concepts and $r$ is spatial relation.

**Failure mode:** Models often answer from language priors without looking — $\text{Acc}_{\text{text-only}} \approx \text{Acc}_{\text{full}}$ indicates insufficient grounding.

---

## Spatial Reasoning — Coordinate Derivation

Represent object locations as normalized bounding boxes $\mathbf{b} = (x_c, y_c, w, h) \in [0,1]^4$.

Spatial relation "left of":

$$
\text{left}(b_1, b_2) \iff x_c^{(1)} + w^{(1)}/2 < x_c^{(2)} - w^{(2)}/2
$$

**Numerical example:**

- Object A: $b_1 = (0.2, 0.5, 0.1, 0.1)$ → right edge at $0.25$
- Object B: $b_2 = (0.6, 0.5, 0.1, 0.1)$ → left edge at $0.55$

$0.25 < 0.55$ → A is left of B ✓

---

## Visual Grounding — Referential Expression

Given expression $E$ and image $I$, find region $\mathbf{b}^*$:

$$
\mathbf{b}^* = \arg\max_{\mathbf{b}} P(\mathbf{b} \mid I, E)
$$

Typically implemented via cross-attention pooling over patch features:

$$
s_j = \text{Attn}(E, \text{patch}_j), \quad \mathbf{b}^* = \text{Soft-Argmax}(\{s_j\}_{j=1}^{N})
$$

**Numerical example:** 4 patches with grounding scores $[0.05, 0.72, 0.15, 0.08]$:

- Peak at patch 1 → model grounds "the red car" to patch 1
- Soft-argmax center: $\approx (0.25, 0.5)$ in normalized coords

---

## Multimodal CoT Prompt Template

```
User: <image> How many triangles are in this figure?
Assistant: Let me analyze step by step.
Step 1: I identify 3 distinct shapes in the image.
Step 2: Shape A (top-left) has 3 sides → triangle.
Step 3: Shape B (center) has 4 sides → not a triangle.
Step 4: Shape C (bottom-right) has 3 sides → triangle.
Therefore, the answer is 2 triangles.
```

Loss encourages matching each step before final answer.

---

## Mathematical Proofs

### Proof: Chain-of-Thought — Factorization and Training

**Step 1 — Direct prediction (no CoT):**

$$
P(a \mid I, Q) = \text{softmax}(\text{Classifier}(\text{Fuse}(I, Q)))
$$

Limited expressivity for multi-step reasoning.

**Step 2 — Introduce reasoning chain $R = (r_1, \ldots, r_T)$:**

$$
P(a \mid I, Q) = \sum_R P(R \mid I, Q) \cdot P(a \mid I, Q, R)
$$

**Step 3 — Autoregressive chain factorization:**

$$
P(R \mid I, Q) = \prod_{t=1}^{T} P(r_t \mid I, Q, r_{1:t-1})
$$

**Step 4 — Marginalization (practical):** Use greedy/beam search over $R$ instead of summing all chains:

$$
P(a \mid I, Q, R^*) \approx P(a \mid I, Q, \hat{R})
$$

where $\hat{R} = \arg\max_R P(R \mid I, Q)$.

**Step 5 — Training loss:**

$$
\mathcal{L} = -\log P(a^* \mid I, Q, R^*) - \lambda \log P(R^* \mid I, Q)
$$

**∎**

#### Numerical Example

"How many red objects?" — $P(r_1) = 0.85$, $P(r_2 \mid r_1) = 0.78$, $P(a=2 \mid R) = 0.92$. Joint $= 0.61$ vs direct $P(a=2) \approx 0.35$ without CoT.

---

## What You'll Build

- CoT prompt template generator
- Spatial relation checker on synthetic bounding boxes
- Grounding score heatmap from attention weights

---

## Next Step

**[07_text_to_image_video/07_text_to_image_video.ipynb](../07_text_to_image_video/07_text_to_image_video.ipynb)**
