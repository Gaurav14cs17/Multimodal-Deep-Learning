# 03 — Overfitting, Underfitting & Regularization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/03_overfitting_underfitting/03_overfitting_underfitting.ipynb)

> **Time:** ~50 minutes | **Difficulty:** Intermediate | **GPU Required:** No

---

## What You'll Learn

- Diagnose train vs val curves — 4 scenarios (good fit, overfit, underfit, double descent)
- Bias-variance decomposition — full derivation
- Why overfitting happens: VC dimension, Rademacher complexity (intuitive)
- Dropout — expected value preservation proof
- Weight decay / L2 — gradient update derivation
- L1 (Lasso) — sparsity property
- Data augmentation — invariance analysis
- Early stopping — implicit regularization
- Label smoothing — cross-entropy effect
- Mixup — ERM proof sketch (Zhang et al.)
- Double descent — interpolation threshold
- Code: overfit → apply fixes → plot results

---

## Key Equations

### 1. Bias-Variance Decomposition

For squared loss and model $\hat{f}(x)$:

$$
\mathbb{E}_{x,y}\left[(y - \hat{f}(x))^2\right] = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

**Step 1:** Write $\hat{f}(x) = \mathbb{E}[\hat{f}(x)] + (\hat{f}(x) - \mathbb{E}[\hat{f}(x)])$

**Step 2:** Add and subtract $\mathbb{E}[y \mid x]$:

$$
(y - \hat{f})^2 = (y - \mathbb{E}[y|x] + \mathbb{E}[y|x] - \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}] - \hat{f})^2
$$

**Step 3:** Cross terms vanish under expectation → three components remain.

### 2. Dropout

$$
\tilde{h} = h \odot \frac{m}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)
$$

**Expected value proof:**

$$
E[\tilde{h}_i] = h_i \cdot (1-p) \cdot \frac{1}{1-p} = h_i
$$

At inference: use $h(1-p)$ or scale weights — equivalent to training expectation.

### 3. L2 Weight Decay

$$
\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2}\lVert W \rVert_F^2
$$

**Gradient update:**

$$
W \leftarrow W - \eta(\nabla_W \mathcal{L} + \lambda W) = (1 - \eta\lambda)W - \eta\nabla_W \mathcal{L}
$$

Each step shrinks weights by factor $(1 - \eta\lambda)$.

### 4. Label Smoothing

$$
y_{smooth} = (1-\epsilon) y + \frac{\epsilon}{K}
$$

For one-hot $y$ with $K$ classes: correct class gets $1-\epsilon + \epsilon/K$, others get $\epsilon/K$.

### 5. Mixup

$$
\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j
$$

$$
\lambda \sim \text{Beta}(\alpha, \alpha)
$$

**ERM view:** Minimizes loss on convex combinations → smoother decision boundary.

---

## Diagnosis ASCII Guide

```
Train loss ↓, Val loss ↓  →  GOOD FIT (keep training)
Train loss ↓, Val loss ↑  →  OVERFIT (add regularization)
Train loss ↑, Val loss ↑  →  UNDERFIT (more capacity)
Val dips, rises, dips again → DOUBLE DESCENT (beyond classical U-curve)
```

---

## Solution Decision Table

| Problem | Solution | When to Use |
|---------|----------|-------------|
| Overfit (val ↑) | Dropout 0.1–0.5 | Large MLPs, transformers |
| Overfit | Weight decay $\lambda=0.01$–$0.1$ | Default for AdamW |
| Overfit | Data augmentation | Vision, audio |
| Overfit | Early stopping | Monitor val loss |
| Overconfident logits | Label smoothing $\epsilon=0.1$ | Classification |
| Small dataset | Mixup $\alpha=0.2$–$1.0$ | Image classification |
| Underfit | More layers, train longer | Both losses high |

---

## Paper References

- Srivastava et al. (2014) — Dropout — [JMLR](https://jmlr.org/papers/v15/srivastava14a.html)
- Zhang et al. (2018) — mixup — [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- Nakkiran et al. (2019) — Deep Double Descent — [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)

**Blog:** [Lilian Weng — Regularization in ML](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
