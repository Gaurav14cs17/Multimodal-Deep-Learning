# 01 — Multimodal Datasets & Data Curation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/06_Datasets_Benchmarks/01_multimodal_datasets/01_multimodal_datasets.ipynb)

> **Time:** ~45 minutes | **GPU Required:** No

---

## What You'll Learn

- Catalog of pretraining, instruction, and evaluation datasets
- How to load and inspect multimodal data (HuggingFace + synthetic fallback)
- Preprocessing: normalization, deduplication, quality filtering
- Dataset scale comparison and distribution analysis

---

## Key Equations

**CLIP filter (pair retention):**

$$
\text{keep}(I, T) \iff \cos(f_I(I), f_T(T)) \geq \tau
$$

**Quality score (weighted):**

$$
Q = \alpha \cdot \text{norm\_length} + \beta \cdot s(I,T) + \gamma \cdot \text{language\_confidence}
$$

**Train mix:**

$$
\mathcal{L}_{\text{total}} = \sum_d w_d \, \mathcal{L}_d, \quad \sum_d w_d = 1
$$

---

## Mathematical Proofs

### Proof: Data Quality Scoring — CLIP Score and Perplexity Filtering

**Step 1 — CLIP similarity score:**

$$
s(I, T) = \cos(f_I(I), f_T(T)) = \frac{f_I(I)^\top f_T(T)}{\lVert f_I(I) \rVert \lVert f_T(T) \rVert}
$$

**Why:** High CLIP score indicates semantic alignment between image and caption.

**Step 2 — Retention rule:**

$$
\text{keep}(I, T) \iff s(I, T) \geq \tau
$$

Typical $\tau \in [0.25, 0.35]$ for LAION filtering.

**Step 3 — Perplexity filter (text quality):**

$$
\text{PPL}(T) = \exp\left(-\frac{1}{L}\sum_{t=1}^{L} \log P(w_t \mid w_{1:t-1})\right)
$$

Keep if $\text{PPL}(T) \leq \tau_{\text{ppl}}$ — removes gibberish captions.

**Step 4 — Combined quality score:**

$$
Q = \alpha \cdot \text{norm\_length}(T) + \beta \cdot s(I,T) + \gamma \cdot \text{language\_confidence}(T)
$$

**∎**

#### Numerical Example

Caption length $= 12$ words (norm $= 0.8$), $s(I,T) = 0.31$, lang conf $= 0.95$. With $\alpha=0.2, \beta=0.6, \gamma=0.2$: $Q = 0.16 + 0.186 + 0.19 = 0.536$. Threshold 0.5 → keep.

---

### Proof: Deduplication — Hashing and Embedding Similarity

**Step 1 — Exact dedup (hashing):**

$$
h(I) = \text{SHA256}(\text{bytes}(I)), \quad \text{keep} \iff h(I) \notin \mathcal{H}_{\text{seen}}
$$

**Step 2 — Near-duplicate detection (embedding):**

$$
\text{dup}(I_i, I_j) \iff \cos(e(I_i), e(I_j)) \geq \tau_{\text{dup}}
$$

Typical $\tau_{\text{dup}} \in [0.95, 0.99]$.

**Step 3 — Cluster-based dedup:** Build graph with edges for pairs above threshold; keep one representative per connected component.

**Step 4 — Complexity:** Hashing is $O(1)$ per sample; embedding dedup with FAISS is $O(N \log N)$ for $N$ samples. **∎**

#### Numerical Example

1M pairs, 5% exact duplicates via hash → 950K unique. Embedding dedup at $\tau=0.97$ removes additional 8% near-dupes → 874K final pairs.

---

## Datasets Covered

| Category | Examples |
|----------|----------|
| Pretrain | LAION-5B, CC3M, CC12M, DataComp, WebLI |
| Instruction | LLaVA-Instruct, ShareGPT4V, SVIT |
| Evaluation | MME, MMMU, MM-Bench, SEED-Bench |

See the [module README](../README.md) for full derivations.
