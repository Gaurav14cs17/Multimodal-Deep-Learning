# 02 — Image Captioning: Image → Text Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/02_image_captioning/02_image_captioning.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

How to generate text descriptions of images — the classic encoder-decoder architecture:

- Image captioning architecture (encoder → cross-attention decoder)
- Causal masking for autoregressive generation
- Decoding strategies: greedy, beam search, top-k, nucleus sampling
- Attention visualization: which image regions generate which words
- Using pretrained BLIP for real captioning

---

## Architecture — Encoder-Decoder with Cross-Attention

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                  Image Captioning Model                         │
  │                                                                 │
  │  ENCODER:                                                       │
  │  Image (224×224) ──► [ViT] ──► Z_img ∈ ℝ^(196 × D)            │
  │                                  │                              │
  │  DECODER (autoregressive):       │ cross-attention              │
  │  ┌───────────────────────────────┼────────────────────┐        │
  │  │ Layer l:                      │                     │        │
  │  │   1. Causal Self-Attention    │                     │        │
  │  │      (text attends to past text only)               │        │
  │  │   2. Cross-Attention ◄────────┘                     │        │
  │  │      (text queries image features)                  │        │
  │  │   3. FFN                                            │        │
  │  └─────────────────────────────────────────────────────┘        │
  │                                                                 │
  │  <start> ──► "A" ──► "cat" ──► "sitting" ──► "on" ──► <end>    │
  │         p₁       p₂       p₃          p₄         p₅            │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Key Equations

### 1. Autoregressive Generation — Conditional Language Modeling

$$
P(\mathbf{y} \mid I) = \prod_{t=1}^{T} P(y_t \mid y_1, \ldots, y_{t-1}, I)
$$

At each step $t$, the model predicts the next token conditioned on all previous tokens and the image.

### 2. Training Loss — Teacher-Forced Cross-Entropy

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_1, \ldots, y_{t-1}, I)
$$

where ground-truth tokens are fed as input at each step (teacher forcing).

**Per-token cross-entropy breakdown:**

$$
\log P_\theta(y_t = w \mid \ldots) = \log \frac{e^{z_w}}{\sum_{v=1}^{V} e^{z_v}} = z_w - \log\sum_{v=1}^{V} e^{z_v}
$$

where $z_w$ is the logit for the correct word $w$, and $V$ is the vocabulary size.

### 3. Causal Masking — Preventing Future Information Leakage

The decoder's self-attention uses a causal mask to ensure each position can only attend to earlier positions:

$$
\text{CausalAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

where the mask $M$ is:

$$
M_{ij} =
\begin{cases}
0 & \text{if } i \geq j \quad \text{(attend)} \\
-\infty & \text{if } i < j \quad \text{(block)}
\end{cases}
$$

```
  Causal mask (5 tokens):
  ┌─────────────────────┐
  │  0    -∞   -∞   -∞  │   ← token 1 sees only itself
  │  0     0   -∞   -∞  │   ← token 2 sees tokens 1-2
  │  0     0    0   -∞  │   ← token 3 sees tokens 1-3
  │  0     0    0    0  │   ← token 4 sees tokens 1-4
  └─────────────────────┘
```

### 4. Cross-Attention in the Decoder

The decoder queries image features at each layer:

$$
\text{CrossAttn}_l = \text{softmax}\!\left(\frac{(H_l W_Q)(Z_{\text{img}} W_K)^\top}{\sqrt{d_k}}\right)(Z_{\text{img}} W_V)
$$

where $H_l$ is the decoder hidden state at layer $l$, and $Z_{\text{img}}$ are the encoder outputs.

---

## Mathematical Proofs

### Proof: Autoregressive Likelihood — Chain Rule Factorization

**Theorem:** The joint probability of caption $\mathbf{y} = (y_1, \ldots, y_T)$ given image $I$ factorizes via the chain rule.

**Step 1 — Joint distribution:**

$$
P(\mathbf{y} \mid I) = P(y_1, y_2, \ldots, y_T \mid I)
$$

**Step 2 — Apply chain rule of probability:**

$$
P(\mathbf{y} \mid I) = P(y_1 \mid I) \cdot P(y_2 \mid y_1, I) \cdots P(y_T \mid y_1, \ldots, y_{T-1}, I)
$$

**Step 3 — Product notation:**

$$
P(\mathbf{y} \mid I) = \prod_{t=1}^{T} P(y_t \mid y_{1:t-1}, I)
$$

**Why:** Causal masking ensures $y_t$ depends only on past tokens — each factor is a valid conditional distribution.

**Step 4 — Training loss (negative log-likelihood):**

$$
\mathcal{L} = -\log P(\mathbf{y} \mid I) = -\sum_{t=1}^{T} \log P(y_t \mid y_{1:t-1}, I)
$$

**∎**

#### Numerical Example

Caption "a cat" ($T=2$), $P(y_1=\text{"a"}\mid I)=0.4$, $P(y_2=\text{"cat"}\mid y_1, I)=0.6$:

$$
P(\mathbf{y}\mid I) = 0.4 \times 0.6 = 0.24, \quad \mathcal{L} = -\log(0.4) - \log(0.6) = 0.916 + 0.511 = 1.427
$$

---

### Proof: Beam Search Approximates MAP Decoding

**Claim:** Beam search with width $B$ approximates $\mathbf{y}^* = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid I)$ by retaining top-$B$ partial hypotheses.

**Step 1 — MAP objective:**

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}} \sum_{t=1}^{T} \log P(y_t \mid y_{1:t-1}, I)
$$

**Step 2 — Greedy is local:** Greedy picks $\arg\max$ at each step — can miss global optimum (e.g., "the cat" vs "a dog").

**Step 3 — Beam search maintains set $\mathcal{B}_t$ with $|\mathcal{B}_t| \leq B$:**

At step $t$, expand each hypothesis in $\mathcal{B}_{t-1}$ with all vocabulary tokens, score by cumulative log-prob, keep top $B$.

**Step 4 — Length normalization prevents bias:**

$$
\text{Score}_{\text{norm}}(\mathbf{y}_{1:t}) = \frac{1}{t^\alpha} \sum_{s=1}^{t} \log P(y_s \mid y_{1:s-1}, I)
$$

**Why:** Without normalization, shorter sequences have higher cumulative probability. **∎**

#### Numerical Example

$B=2$, two-step vocab $\{\text{"a"}, \text{"the"}\}$. Step 1: keep "a" (0.6), "the" (0.5). Step 2 from "a": "a cat" (0.6×0.8=0.48), "a dog" (0.6×0.1=0.06). Beam selects "a cat" — greedy would agree here, but beam preserves "the cat" if step-2 prob is higher globally.

---

### Proof: BLEU — Modified n-gram Precision with Brevity Penalty

**Step 1 — Modified precision for n-grams:**

$$
p_n = \frac{\sum_{\text{n-gram} \in \text{hyp}} \min(\text{Count}_{\text{hyp}}(\text{n-gram}), \text{Count}_{\text{ref}}(\text{n-gram}))}{\sum_{\text{n-gram} \in \text{hyp}} \text{Count}_{\text{hyp}}(\text{n-gram})}
$$

**Why:** Clips counts so repeating "the the the" doesn't inflate precision.

**Step 2 — Geometric mean across orders:**

$$
\text{BLEU-}N = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} \frac{1}{N} \log p_n\right)
$$

**Step 3 — Brevity penalty** ($c$ = candidate length, $r$ = reference length):

$$
\text{BP} =
\begin{cases}
1 & \text{if } c > r \\
e^{1 - r/c} & \text{if } c \leq r
\end{cases}
$$

**Why:** Penalizes overly short outputs that game n-gram precision. **∎**

#### Numerical Example

Hypothesis: "the cat the cat" ($c=4$), reference: "the cat is on the mat" ($r=6$).

- 1-gram: matched 4/4 clipped → $p_1 = 1.0$
- 2-gram: "the cat" appears twice in both → $p_2 = 2/3 \approx 0.667$
- BP: $c < r$ → $e^{1-6/4} = e^{-0.5} \approx 0.607$
- BLEU-2 $\approx 0.607 \times \sqrt{1.0 \times 0.667} \approx 0.496$

---

## Decoding Strategies — Detailed Comparison

### Greedy Decoding

$$
y_t = \arg\max_w P(w \mid y_1, \ldots, y_{t-1}, I)
$$

Always picks the highest-probability token. Fast but can get stuck in repetitive or suboptimal sequences.

### Beam Search (width $B$)

Maintains $B$ candidate sequences at each step:

$$
\text{Score}(\mathbf{y}_{1:t}) = \sum_{s=1}^{t} \log P(y_s \mid y_1, \ldots, y_{s-1}, I)
$$

With **length normalization** to avoid bias toward short sequences:

$$
\text{Score}_{\text{norm}} = \frac{1}{t^\alpha} \sum_{s=1}^{t} \log P(y_s \mid y_1, \ldots, y_{s-1}, I)
$$

where $\alpha \in [0.6, 1.0]$ is a length penalty.

### Top-$k$ Sampling

Sample from the top $k$ tokens by probability:

$$
P'(w) =
\begin{cases}
P(w) / Z_k & \text{if } w \in \text{top-}k \\
0 & \text{otherwise}
\end{cases}
$$

where $Z_k = \sum_{w \in \text{top-}k} P(w)$ is the re-normalization constant.

### Nucleus (Top-$p$) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds $p$:

$$
V_p = \min\!\left\{S \subseteq V : \sum_{w \in S} P(w) \geq p\right\}
$$

$$
P'(w) =
\begin{cases}
P(w) / \sum_{w' \in V_p} P(w') & \text{if } w \in V_p \\
0 & \text{otherwise}
\end{cases}
$$

### Comparison Table

| Strategy | Quality | Diversity | Speed | When to Use |
|----------|---------|-----------|-------|-------------|
| Greedy | Low | None | Fastest | Debugging, deterministic output |
| Beam ($B$=5) | High | Low | Medium | Evaluation benchmarks |
| Top-$k$ ($k$=50) | Good | High | Fast | Creative text generation |
| Nucleus ($p$=0.9) | Good | High | Fast | Production generation (recommended) |
| Beam + Nucleus | Highest | Medium | Slow | Research papers |

---

## Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)

Precision of $n$-grams between generated and reference captions:

$$
\text{BLEU-}N = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} \frac{1}{N} \log p_n\right)
$$

where $p_n = \frac{\text{matched } n\text{-grams}}{\text{total generated } n\text{-grams}}$, and the brevity penalty:

$$
\text{BP} =
\begin{cases}
1 & \text{if } c > r \\
e^{1 - r/c} & \text{if } c \leq r
\end{cases}
$$

($c$ = generated length, $r$ = reference length).

### CIDEr (Consensus-based Image Description Evaluation)

TF-IDF weighted $n$-gram similarity:

$$
\text{CIDEr}_n = \frac{1}{M}\sum_{j=1}^{M} \frac{\mathbf{g}_n(\hat{y}) \cdot \mathbf{g}_n(y_j)}{\lVert \mathbf{g}_n(\hat{y}) \rVert \cdot \lVert \mathbf{g}_n(y_j) \rVert}
$$

$$
\text{CIDEr} = \sum_{n=1}^{4} w_n \cdot \text{CIDEr}_n
$$

where $\mathbf{g}_n$ are TF-IDF vectors for $n$-grams, summed over $M$ reference captions.

---

## What You'll Build

- Encoder-decoder captioning model from scratch
- Training loop with teacher forcing
- All 4 decoding strategies
- Attention visualization: which image regions generate which words
- BLIP inference for real captioning

---

## Prerequisites

- Module 01 notebooks (ViT encoder, cross-attention)
- Understanding of autoregressive generation
- Familiarity with softmax and cross-entropy loss

---

## 🔬 Worked Examples in the Notebook

### Example 1: Token-Level Cross-Entropy Trace
Compute captioning loss step-by-step for "[BOS] a cute cat [EOS]":
- Per-token probability and loss breakdown
- Identify hardest token (largest gradient signal)
- Perplexity calculation and teacher forcing vs autoregressive gap

### Example 2: Beam Search Implementation
Full beam search decoder with comparison to greedy:
- Beam search with B=2, 3, 5 on the same image
- Length-normalized scoring
- Top-k vs nucleus sampling visualization

### Example 3: BLEU Score — Hand Computation
Compute BLEU-4 step-by-step for three quality levels:
- Good caption (high n-gram overlap): BLEU ≈ 0.45
- Mediocre caption (partial overlap): BLEU ≈ 0.05
- Wrong caption (no overlap): BLEU ≈ 0.0

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/02_Vision_Language_Models/02_image_captioning/02_image_captioning.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| BLIP-2 Framework (`../../assets/paper_figures/blip2_framework.png`) | Li et al. — [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) | 2023 | Q-Former bridges frozen ViT + frozen LLM |
| BLIP Unified Architecture | Li et al. — [arXiv:2201.12086](https://arxiv.org/abs/2201.12086) | 2022 | Shared ViT encoder, 3 objectives (ITC/ITM/LM), CapFilt |

### Additional Papers Covered

- **Show, Attend and Tell** (Xu et al., 2015) — Attention-based image captioning
- **BLIP-2** (Li et al., 2023) — Q-Former bridges frozen ViT + frozen LLM
- **CoCa** (Yu et al., 2022) — Contrastive + captioning in single model

---

## Next Step

**[03_visual_question_answering](../03_visual_question_answering/)** — Image + Question → Answer
