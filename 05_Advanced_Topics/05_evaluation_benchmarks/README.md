# 05 — Evaluation & Benchmarks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb)

> **Time:** ~40 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

How to properly evaluate multimodal models:

- MMMU, MME, MM-Bench methodology
- Zero-shot vs finetuned evaluation protocols
- Statistical significance and leaderboard pitfalls
- Implement Recall@K and VQA accuracy

---

## VQA Accuracy — Derivation

### Step 1: Prediction

Model outputs answer $\hat{a}$ from vocabulary $\mathcal{A}$ with $|\mathcal{A}| = K$:

$$
\hat{a} = \arg\max_{a \in \mathcal{A}} P_\theta(a \mid I, Q)
$$

### Step 2: Accuracy

For $M$ test samples with ground truth $a_i^*$:

$$
\text{Acc} = \frac{1}{M}\sum_{i=1}^{M}\mathbb{1}[\hat{a}_i = a_i^*]
$$

### Numerical Example

$M=5$ predictions: `[cat, dog, cat, red, cat]`, ground truth: `[cat, cat, cat, red, dog]`

| $i$ | Pred | GT | Match |
|-----|------|-----|-------|
| 1 | cat | cat | ✓ |
| 2 | dog | cat | ✗ |
| 3 | cat | cat | ✓ |
| 4 | red | red | ✓ |
| 5 | cat | dog | ✗ |

$$
\text{Acc} = \frac{3}{5} = 0.60 = 60\%
$$

---

## Recall@K for Retrieval — Full Derivation

### Step 1: Similarity Ranking

For query $q_i$, rank gallery items by $s_{ij} = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j$.

### Step 2: Recall@K

$$
\text{R@}K = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[\exists j \in \text{top-}K(i): j = i\right]
$$

For image→text retrieval, $j = i$ means correct caption in top $K$.

### Numerical Example ($N=3$)

Similarity matrix (image → text):

$$
\mathbf{S} = \begin{pmatrix} 0.95 & 0.12 & 0.08 \\ 0.10 & 0.88 & 0.15 \\ 0.05 & 0.20 & 0.91 \end{pmatrix}
$$

Ranks: image 0 → text 0 (rank 1), image 1 → text 1 (rank 1), image 2 → text 2 (rank 1).

$$
\text{R@1} = \frac{3}{3} = 1.0, \quad \text{R@5} = 1.0
$$

If image 0 similarities were `[0.12, 0.95, 0.08]`:

- Correct text 0 ranks 2nd → R@1 = 0, R@2 = 1 for that query
- Overall R@1 = 2/3 ≈ 0.667

---

## MMMU — Multi-Discipline Multimodal Understanding

MMMU evaluates college-level reasoning across 6 disciplines, 30 subjects, 11K questions.

**Scoring:** Multiple choice — exact match:

$$
\text{MMMU-Acc} = \frac{\#\{\text{correct MC answers}\}}{\#\{\text{total questions}\}}
$$

**Per-subject breakdown** reveals whether model excels at diagrams (Math) vs photographs (Art).

---

## MME — Perception vs Cognition Split

MME decomposes evaluation:

| Category | Examples | Metric |
|----------|----------|--------|
| Perception | OCR, color, count | Yes/No accuracy |
| Cognition | reasoning, math | Yes/No accuracy |

$$
\text{MME-Score} = \sum_{\text{task}} \text{Acc}_{\text{task}} \times 100
$$

**Numerical example:** 14 subtasks, accuracies $[0.9, 0.85, 0.7, \ldots]$:

$$
\text{MME} = (0.9 + 0.85 + 0.7 + \cdots) \times 100
$$

Typical strong model: MME ≈ 1800–2000 (max ≈ 2800).

---

## MM-Bench — Circular Evaluation

To prevent position bias in multiple choice, MM-Bench uses **circular evaluation**: each question is evaluated with all answer permutations.

$$
\text{Acc}_{\text{circular}} = \frac{1}{4}\sum_{p=1}^{4}\mathbb{1}[\text{correct under permutation } p]
$$

All 4 permutations must be correct for full credit — stricter than standard MC accuracy.

---

## Statistical Significance

For accuracy $\hat{p}$ on $M$ samples, 95% confidence interval (Wilson):

$$
\text{CI} = \hat{p} \pm 1.96\sqrt{\frac{\hat{p}(1-\hat{p})}{M}}
$$

**Numerical example:** $\hat{p} = 0.72$, $M = 100$:

$$
\text{SE} = \sqrt{\frac{0.72 \times 0.28}{100}} = 0.045, \quad \text{CI} = [0.63, 0.81]
$$

Difference of 2% between models on 100 samples is **not significant**.

---

## Mathematical Proofs

### Proof: Recall@K — Full Metric Derivation

**Step 1 — Similarity ranking:** For query $i$, compute $s_{ij} = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j$ for all $j$.

**Step 2 — Rank of ground truth:** $\text{rank}(i,i) = 1 + \sum_{j \neq i} \mathbb{1}[s_{ij} > s_{ii}]$.

**Step 3 — Recall@K definition:**

$$
\text{R@}K = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\text{rank}(i,i) \leq K]
$$

**Why:** Measures whether the correct match appears in top-$K$ — standard retrieval metric.

**Step 4 — R-Precision variant:** When $K$ equals number of relevant items per query, R-Precision = precision at fixed recall. **∎**

#### Numerical Example

From matrix $\mathbf{S}$ with diagonal dominant: all ranks $= 1$, R@1 $= 3/3 = 1.0$. If row 0 were $[0.12, 0.95, 0.08]$: rank of match 0 is 2, R@1 $= 0$, R@2 $= 1$.

---

### Proof: VQA Accuracy — Hard vs Soft Evaluation

**Step 1 — Hard accuracy:**

$$
\text{Acc}_{\text{hard}} = \frac{1}{M}\sum_{i=1}^{M}\mathbb{1}[\hat{a}_i = a_i^*]
$$

**Step 2 — VQA v2 soft accuracy:** For $K$ annotators, answer accepted if $\geq \lceil K/2 \rceil$ agree with prediction:

$$
\text{Acc}_{\text{soft}} = \frac{1}{M}\sum_{i=1}^{M}\min\left(\frac{\#\{\text{annotators matching } \hat{a}_i\}}{K}, 1\right)
$$

**Step 3 — Why soft:** Accounts for legitimate answer ambiguity (e.g., "couch" vs "sofa"). **∎**

#### Numerical Example

10 annotators, 7 say "cat", model predicts "cat": soft acc contribution $= \min(7/10, 1) = 0.7$. Hard acc $= 1$.

---

## What You'll Build

- VQA accuracy and Recall@K functions
- Synthetic benchmark evaluator
- Confidence interval calculator

---

## Next Step

**[06_multimodal_reasoning/06_multimodal_reasoning.ipynb](../06_multimodal_reasoning/06_multimodal_reasoning.ipynb)**
