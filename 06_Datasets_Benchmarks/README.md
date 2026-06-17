# Module 06: Datasets & Benchmarks

> **Time:** 1–2 hours | **Notebooks:** 1 | **Difficulty:** Intermediate

| Notebook | Open in Colab |
|----------|---------------|
| `01_multimodal_datasets/01_multimodal_datasets.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/06_Datasets_Benchmarks/01_multimodal_datasets/01_multimodal_datasets.ipynb) |

---

## Why This Module Matters

Multimodal models are only as good as the data they see. This module covers **what datasets exist**, **how they are curated**, and **how to build preprocessing pipelines** before training or evaluation.

```
  Raw Web Data  ──►  Filtering  ──►  Dedup  ──►  Quality Score  ──►  Training Mix
       │                │              │              │                    │
   LAION/CC/WebLI    CLIP score     MinHash      length/OCR         curriculum
```

---

## Pre-training Datasets

| Dataset | Scale | Modality | Key Property |
|---------|-------|----------|--------------|
| **LAION-5B** | 5.85B pairs | image–text | CLIP-filtered web crawl |
| **CC3M** | 3.3M | image–text | Cleaner alt-text captions |
| **CC12M** | 12.4M | image–text | Larger but noisier |
| **DataComp** | Variable | image–text | Systematic filtering benchmarks |
| **WebLI** | 10B+ | image–text | Google-scale multilingual web data |

### CLIP-Score Filtering (LAION-style)

For image $I$ and text $T$, keep pairs where cosine similarity exceeds threshold $\tau$:

$$
s(I, T) = \frac{f_I(I)^\top f_T(T)}{\|f_I(I)\| \|f_T(T)\|} \geq \tau
$$

Typical $\tau \in [0.28, 0.35]$ for LAION subsets. This removes mismatched alt-text at scale.

---

## Instruction-Tuning Datasets

| Dataset | Size | Format | Used By |
|---------|------|--------|---------|
| **LLaVA-Instruct** | 150K | `<image>` + conversation | LLaVA stage 2 |
| **ShareGPT4V** | 100K+ | GPT-4V dialogues | Open MLLM tuning |
| **SVIT** | 4.8M | Synthetic VLM instructions | Scalable instruction data |

Instruction data quality dominates chat capability more than pretrain scale alone.

---

## Evaluation Benchmarks

| Benchmark | What It Measures | Metric |
|-----------|------------------|--------|
| **MME** | Perception + cognition | Per-category accuracy sum |
| **MMMU** | Multi-discipline reasoning | Multiple-choice accuracy |
| **MM-Bench** | Structured LVLM skills | Circular-eval accuracy |
| **SEED-Bench** | 19 capability dimensions | Multiple-choice accuracy |

### VQA Accuracy

For $N$ questions with answers $\{y_i\}$ and predictions $\{\hat{y}_i\}$:

$$
\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
$$

---

## Data Curation Techniques

### 1. Deduplication (MinHash / SimHash)

Approximate Jaccard similarity between documents $A, B$:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

MinHash estimates $J(A,B)$ in $O(1)$ per pair after sketching — critical at billion-pair scale.

### 2. Quality Scoring

Composite score for caption $c$:

$$
Q(c) = w_1 \cdot f_{\text{len}}(c) + w_2 \cdot f_{\text{clip}}(I, c) + w_3 \cdot f_{\text{lang}}(c)
$$

Keep pairs with $Q(c) \geq \theta$.

### 3. Curriculum / Mixing

Training loss over mixed sources:

$$
\mathcal{L} = \sum_{d=1}^{D} w_d \cdot \mathbb{E}_{(x,y) \sim \mathcal{D}_d}[\ell(x, y)], \quad \sum_d w_d = 1
$$

Start with high-$Q$ data; gradually add hard/noisy sources.

---

## Notebook: `01_multimodal_datasets`

Hands-on coverage:
1. Load synthetic + optional HuggingFace samples
2. Caption length / source distribution plots
3. Preprocessing pipeline (normalize, truncate, dedup)
4. Quality scoring and filtering
5. Log-scale dataset size comparison chart

---

## Next Step

Return to training with curated data:

**[03_Training_Strategies/04_multimodal_alignment/04_multimodal_alignment.ipynb](../03_Training_Strategies/04_multimodal_alignment/04_multimodal_alignment.ipynb)**
