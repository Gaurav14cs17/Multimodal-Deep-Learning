# 01 — Vanishing & Exploding Gradients

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/01_gradient_problems/01_gradient_problems.ipynb)

> **Time:** ~50 minutes | **Difficulty:** Intermediate–Advanced | **GPU Required:** No

---

## What You'll Learn

- Full chain-rule derivation of gradient flow through $L$ layers
- Mathematical conditions for vanishing ($\lVert J_l \rVert < 1$) vs exploding ($\lVert J_l \rVert > 1$) gradients
- Numerical trace through 10 layers with concrete numbers
- Gradient clipping formula and when it helps
- Residual connections — proof that $\frac{\partial h_L}{\partial h_l}$ includes identity term
- Layer Normalization / BatchNorm — Jacobian stabilization
- Xavier/Glorot and He/Kaiming initialization — variance preservation derivations
- Mixed-precision gradient scaling
- Decision table: symptom → fix → priority

---

## Key Equations

### 1. Chain Rule Through $L$ Layers

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \left(\prod_{l=1}^{L} \frac{\partial h_l}{\partial h_{l-1}}\right) \frac{\partial \mathcal{L}}{\partial h_L}
$$

**What each term does:**
- $\frac{\partial h_l}{\partial h_{l-1}}$ — Jacobian $J_l$ of layer $l$
- Product over $L$ layers — exponential scaling in depth
- If $\lVert J_l \rVert < 1$ for all $l$ → product → 0 (vanishing)
- If $\lVert J_l \rVert > 1$ for all $l$ → product → ∞ (exploding)

### 2. Vanishing — Full Step-by-Step Numerical Example

**Setup:** 10 layers, $\lVert J_l \rVert = 0.5$, $\lVert \partial \mathcal{L}/\partial h_{10} \rVert = 1.0$

**Step 1:** Layer 10 gradient norm = 1.0

**Step 2:** Layer 9: $1.0 \times 0.5 = 0.5$

**Step 3:** Layer 8: $0.5 \times 0.5 = 0.25$

**Step 4:** Continue: Layer $l$ norm = $0.5^{10-l}$

**Step 5:** Layer 1: $0.5^9 = 0.00195$ — **99.8% of gradient lost**

### 3. Gradient Clipping

$$
g \leftarrow g \cdot \frac{\text{max\_norm}}{\max(\lVert g \rVert, \text{max\_norm})}
$$

Preserves direction; caps magnitude at `max_norm` (typically 1.0).

### 4. Residual Connection — Gradient Flow Proof

$$
h_l = F(h_{l-1}) + h_{l-1}
$$

**Step 1:** Apply chain rule:

$$
\frac{\partial h_L}{\partial h_l} = \frac{\partial h_L}{\partial h_{l+1}} \cdot \frac{\partial h_{l+1}}{\partial h_l}
$$

**Step 2:** Compute $\frac{\partial h_{l+1}}{\partial h_l}$:

$$
\frac{\partial h_{l+1}}{\partial h_l} = \frac{\partial F(h_l)}{\partial h_l} + I
$$

**Step 3:** Recurse — at minimum, identity paths carry gradient $\frac{\partial \mathcal{L}}{\partial h_L}$ directly to every layer.

### 5. Layer Normalization

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$.

**Effect:** Normalizes activation scale → Jacobian singular values cluster near 1.

### 6. Xavier (Glorot) Initialization

For linear layer with $n_{in}$ inputs, $n_{out}$ outputs:

$$
\text{Var}(W_{ij}) = \frac{2}{n_{in} + n_{out}}
$$

**Derivation sketch:** Forward pass variance $\text{Var}(h_l) \approx \text{Var}(h_{l-1})$ when $n_{in} \cdot \text{Var}(W) \approx 1$ and backward pass similarly with $n_{out}$.

### 7. He (Kaiming) Initialization

For ReLU activations (variance halved by ReLU):

$$
\text{Var}(W_{ij}) = \frac{2}{n_{in}}
$$

---

## ASCII Diagram — Gradient Flow

```
Plain Deep Network:
  h0 --[W1]--> h1 --[W2]--> ... --[WL]--> L
  grad:  dL/dh0 <-- dL/dh1 <-- ... <-- dL/dhL
         (product shrinks each step if ||J||<1)

Residual Network:
  h0 --+--[F1]--+-- h1 --+--[F2]--+-- h2 ...
       |         |        |         |
       +---------+        +---------+
  grad: skip paths (identity) preserve magnitude
```

---

## Decision Table

| Symptom | Likely Cause | Solution | When to Use |
|---------|--------------|----------|-------------|
| Early layer grads ≈ 0 | Vanishing Jacobians | Residual + He init + LN | Transformers, ResNets, depth > 20 |
| Loss spikes, grad > 100 | Exploding | `clip_grad_norm_(max_norm=1.0)` | RNNs, high LR |
| Activations drift | Internal covariate shift | LayerNorm / BatchNorm | Before each nonlinearity |
| FP16 no learning | Grad underflow | Loss scaling (AMP) | Mixed precision training |
| Tanh/sigmoid nets | Saturated derivatives | Xavier init, switch to ReLU | Legacy architectures |

---

## Paper References

- Glorot & Bengio (2010) — [arXiv:1006.2785](https://arxiv.org/abs/1006.2785)
- He et al. (2015) — [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- Ba et al. (2016) — Layer Normalization — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

**Blog:** [Lilian Weng — Why ResNet Works](https://lilianweng.github.io/posts/2017-06-08-overview/)
