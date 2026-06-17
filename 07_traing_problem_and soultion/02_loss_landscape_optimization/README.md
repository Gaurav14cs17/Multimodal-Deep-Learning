# 02 — Loss Landscape, Learning Rate & Optimizer Problems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/07_traing_problem_and%20soultion/02_loss_landscape_optimization/02_loss_landscape_optimization.ipynb)

> **Time:** ~55 minutes | **Difficulty:** Intermediate–Advanced | **GPU Required:** No

---

## What You'll Learn

- Loss landscape geometry: saddle points, sharp vs flat minima
- Why LR too high diverges — proof on quadratic bowl
- Why LR too low causes slow/stuck training
- LR warmup — gradient variance at initialization
- SGD + Momentum — effective learning rate analysis
- Adam — full update equations + bias correction proof
- AdamW vs Adam — why weight decay ≠ L2 regularization
- LAMB/LARS for large-batch training
- LR schedulers: cosine, warmup+cosine, one-cycle, step decay
- Numerical comparison: SGD vs Adam vs AdamW loss curves

---

## Key Equations

### 1. Quadratic Divergence Condition

For $\mathcal{L}(w) = \frac{\lambda}{2} w^2$, gradient descent:

$$
w_{t+1} = w_t - \eta \lambda w_t = (1 - \eta\lambda) w_t
$$

**Step 1:** Stability requires $\lvert 1 - \eta\lambda \rvert < 1$

**Step 2:** Upper bound: $\eta < 2/\lambda$

**Step 3:** Oscillation when $\eta\lambda > 1$ (sign alternates)

**Numerical example:** $\lambda = 1$, $\eta = 3$ → $w_1 = -2w_0$ → diverges.

### 2. SGD with Momentum

$$
v_t = \beta v_{t-1} + g_t
$$

$$
w_t = w_{t-1} - \eta v_t
$$

**Effective LR:** In consistent gradient direction, $v_t \approx g_t / (1-\beta)$ → effective step $\approx \eta / (1-\beta)$.

### 3. Adam Updates

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
w_t = w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

### 4. Bias Correction Proof

**Step 1:** Assume $E[g_t] = g$ (constant). Then:

$$
E[m_t] = (1-\beta_1) g \sum_{i=0}^{t-1} \beta_1^i = (1-\beta_1^t) g
$$

**Step 2:** Raw $m_t$ is biased toward 0 at early steps.

**Step 3:** Dividing by $(1-\beta_1^t)$ gives $E[\hat{m}_t] = g$ — unbiased.

### 5. AdamW (Decoupled Weight Decay)

$$
w_t = w_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda w_{t-1}\right)
$$

L2 in Adam adds $\lambda w$ to **gradient** before adaptive scaling — incorrect coupling. AdamW applies decay directly on weights.

### 6. Cosine Annealing

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

---

## ASCII Diagram — Loss Landscape

```
        Loss
         |     sharp min (poor generalization)
         |      \_/
         |   ___/   \___  flat min (good generalization)
         |__/             \___
         +------------------------> parameter w

Saddle point (common in high-D):
         |    \    /
         |     \  /
         |      \/   <-- gradient=0 but not minimum
         |      /\
         |     /  \
```

---

## Scheduler Decision Table

| Scheduler | When to Use | Typical Setting |
|-----------|-------------|-----------------|
| Constant | Baseline, small models | Fixed $\eta$ |
| Step decay | ImageNet CNNs | $\gamma=0.1$ every 30 epochs |
| Cosine | Transformers, CLIP | $\eta_{min}=10^{-5}$ |
| Warmup + cosine | Large models + Adam | 1–5% warmup steps |
| One-cycle | Fast training (Smith) | Max LR in middle of cycle |

---

## Paper References

- Kingma & Ba (2015) — Adam — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Loshchilov & Hutter (2019) — AdamW — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Smith (2017) — Cyclical LR — [arXiv:1506.01186](https://arxiv.org/abs/1506.01186)
- You et al. (2019) — LARS/LAMB — [arXiv:1904.00962](https://arxiv.org/abs/1904.00962)

**Blog:** [Lilian Weng — Large Batch Training](https://lilianweng.github.io/posts/2021-12-05-large-batch/)
