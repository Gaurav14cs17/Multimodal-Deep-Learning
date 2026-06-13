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

## Next Step

**[03_visual_question_answering](../03_visual_question_answering/)** — Image + Question → Answer
