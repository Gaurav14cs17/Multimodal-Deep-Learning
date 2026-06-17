# 06 — Tokenization & Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/06_tokenization_embeddings/06_tokenization_embeddings.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

Before encoders process data, raw inputs must become discrete tokens or patches:

- BPE, WordPiece, SentencePiece algorithms (step-by-step)
- Patch embedding for vision (ViT)
- Embedding lookup mathematics
- Hands-on tokenization and patchification

---

## Byte Pair Encoding (BPE) — Full Algorithm

### Step 1: Initialize Vocabulary

Start with all single characters (bytes) in the corpus:

$$
\mathcal{V}_0 = \{\text{a, b, c, \ldots, \text{space}, \ldots}\}
$$

### Step 2: Count Adjacent Pairs

Scan corpus and count frequency of every adjacent symbol pair $(x, y)$:

$$
\text{count}(x, y) = \#\{(x, y) \text{ appears adjacently in corpus}\}
$$

### Step 3: Merge Most Frequent Pair

$$
(x^*, y^*) = \arg\max_{(x,y)} \text{count}(x, y)
$$

Add merged token $xy$ to vocabulary: $\mathcal{V}_{t+1} = \mathcal{V}_t \cup \{xy\}$.

Replace all occurrences of $xy$ in corpus with the new token.

### Step 4: Repeat

Repeat Steps 2–3 for $K$ merge operations (typically $K = 30{,}000$ for GPT-2).

### Numerical Worked Example

**Corpus:** `"low low lower lowest"` (4 words)

**Iteration 0:** Vocab = `{l, o, w, e, r, s, t}` + space

| Pair | Count |
|------|-------|
| (l, o) | 4 |
| (o, w) | 4 |
| (w, space) | 2 |
| (w, e) | 2 |

**Merge 1:** `(l, o) → lo`. Corpus: `"lo w lo w lo wer lo west"`

**Merge 2:** `(lo, w) → low`. Corpus: `"low low low er low est"`

**Merge 3:** `(low, space) → low_` (space token). Continue until vocab size target.

**Final tokenization of "lower":** `["low", "er"]` (2 tokens instead of 5 characters).

---

## WordPiece — Likelihood-Based Merging

WordPiece (BERT) chooses merges maximizing training data likelihood:

$$
\text{score}(x, y) = \frac{\text{count}(xy)}{\text{count}(x) \cdot \text{count}(y)}
$$

**Numerical example:**

- $\text{count}(e) = 1000$, $\text{count}(r) = 800$, $\text{count}(er) = 600$

$$
\text{score}(e, r) = \frac{600}{1000 \times 800} = 0.00075
$$

Higher score → stronger association → prefer merge. WordPiece prefixes continuations with `##`: `"playing" → ["play", "##ing"]`.

---

## Patch Embedding (Vision) — Derivation

### Step 1: Partition Image

Image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ into $N = HW/P^2$ patches:

$$
\mathbf{x}_p^i \in \mathbb{R}^{P^2 C}, \quad i = 1, \ldots, N
$$

### Step 2: Linear Projection

$$
\mathbf{z}_i = \mathbf{x}_p^i \mathbf{E} + \mathbf{b}, \quad \mathbf{E} \in \mathbb{R}^{P^2 C \times D}
$$

### Step 3: Add Positional Encoding

$$
\mathbf{z}_i' = \mathbf{z}_i + \mathbf{e}_{\text{pos}}^i
$$

### Numerical Example

$H = W = 32$, $P = 4$, $C = 3$, $D = 128$:

$$
N = \frac{32 \times 32}{16} = 64 \text{ patches}, \quad P^2 C = 48
$$

Projection matrix $\mathbf{E}$: $48 \times 128 = 6{,}144$ parameters.

Patch 0 flattened (example): $\mathbf{x}_p^0 = [0.2, 0.5, \ldots] \in \mathbb{R}^{48}$

$$
\mathbf{z}_0 = \mathbf{x}_p^0 \mathbf{E} \in \mathbb{R}^{128}
$$

If $\|\mathbf{x}_p^0\| = 1.2$ and typical row of $\mathbf{E}$ has norm 0.3, then $\|\mathbf{z}_0\| \approx 0.3 \times 1.2 \times \sqrt{48} \approx 2.5$ (before LayerNorm).

---

## Embedding Lookup — Mathematical Form

Token ID $w_t \in \{0, \ldots, V-1\}$:

$$
\mathbf{e}_t = \mathbf{E}_{\text{embed}}[w_t] = \mathbf{E}_{\text{embed}}^\top \mathbf{o}_t
$$

where $\mathbf{o}_t$ is one-hot: $(\mathbf{o}_t)_j = \mathbb{1}[w_t = j]$.

**Numerical example:** $V = 5$, $D = 3$, $w_t = 2$:

$$
\mathbf{o}_t = [0, 0, 1, 0, 0]^\top, \quad
\mathbf{E}_{\text{embed}} = \begin{pmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \\ 0.7 & 0.8 \\ 0.9 & 1.0 \end{pmatrix}
$$

$$
\mathbf{e}_t = [0.5, 0.6]^\top \quad \text{(row 2 of embedding matrix)}
$$

Parameters: $V \times D = 5 \times 3 = 15$.

---

## SentencePiece — Unigram Language Model

SentencePiece treats the corpus as a sequence of subword units and finds the vocabulary $\mathcal{V}$ maximizing:

$$
\mathcal{L}(\mathcal{V}) = \sum_{x \in \mathcal{D}} \log P(x \mid \mathcal{V})
$$

where $P(x \mid \mathcal{V}) = \prod_i P(u_i)$ and each $u_i \in \mathcal{V}$.

**Key advantage:** No pre-tokenization — handles multilingual text and whitespace as explicit symbols (`▁` for space).

---

## What You'll Build

- BPE merge simulation on toy corpus
- `PatchEmbed` module (Conv2d or linear unfold)
- Side-by-side: token IDs → embeddings vs patches → embeddings

---

## Next Step

**[02_Vision_Language_Models/01_clip_from_scratch/01_clip_from_scratch.ipynb](../../02_Vision_Language_Models/01_clip_from_scratch/01_clip_from_scratch.ipynb)**
