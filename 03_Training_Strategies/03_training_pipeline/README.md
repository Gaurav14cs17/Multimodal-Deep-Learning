# 03 — Full Training Pipeline + Multi-Stage Pipelines (SFT → RL → Merge)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline/03_training_pipeline.ipynb)

> **Time:** ~60 minutes | **Difficulty:** Advanced | **GPU Required:** No (CPU-friendly)

---

## What You'll Learn

This is the **most comprehensive notebook** in the course — it covers both the practical training loop AND the theory behind modern 3-stage pipelines.

### Part 1: Practical Training Loop
- Complete production-quality training loop with all best practices
- Data pipeline (loading, augmentation, batching)
- AdamW optimizer + cosine LR with warmup
- Gradient accumulation (simulate 1024 batch on a single GPU)
- Mixed precision, gradient clipping, checkpointing
- Live training visualization

### Part 2: Real-World Multi-Stage Pipelines (SFT → RL → Merge)
- **Stage 1: SFT** — cross-entropy loss, knowledge distillation with $T^2$ scaling proof
- **Stage 2: RL** — three algorithms with full math:
  - RLHF (PPO + reward model + GAE)
  - DPO (full derivation from RLHF → closed-form → $Z(x)$ cancellation)
  - GRPO (group-relative advantages + verifiable rewards + numerical example)
- **Stage 3: Weight Merging** — Linear soup, Task arithmetic, TIES merging
- Two end-to-end examples with full pipeline diagrams:
  - LLaVA-style (SFT → Instruction Tuning → DPO → Merge)
  - OCR-Specialized VLM (SFT+KD → GRPO → Model Souping)

---

## Part 1: Practical Training Loop — Key Equations

### 1. AdamW Optimizer

AdamW decouples weight decay from the gradient update (unlike Adam with L2 regularization):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

where $\lambda$ is the **decoupled weight decay** (typically $0.01$), not part of the gradient.

| Hyperparameter | Symbol | Typical Value |
|---------------|--------|-------------|
| Learning rate | $\eta$ | $1 \times 10^{-4}$ to $5 \times 10^{-5}$ |
| $\beta_1$ (momentum) | $\beta_1$ | 0.9 |
| $\beta_2$ (RMSProp) | $\beta_2$ | 0.999 |
| Epsilon | $\epsilon$ | $10^{-8}$ |
| Weight decay | $\lambda$ | 0.01 |

### 2. Cosine Learning Rate Schedule with Warmup

$$
\eta(t) =
\begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warm}}} & \text{if } t < T_{\text{warm}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{t - T_{\text{warm}}}{T - T_{\text{warm}}} \pi\right)\right) & \text{otherwise}
\end{cases}
$$

```
  Learning rate over training:

  η_max ┤    ╱──────────────╲
        │   ╱                 ╲
        │  ╱                    ╲
        │ ╱                       ╲
        │╱                          ╲
  η_min ┤                            ╲_____
        └──┴──────────────────────────┴──
         0  T_warm                      T
          Linear      Cosine decay
          warmup
```

**Why warmup?** At the start of training, Adam's $v_t$ estimates are noisy (initialized to 0). Large learning rates cause divergence. Warmup gives the optimizer time to calibrate.

### 3. Gradient Accumulation — Simulate Large Batches

$$
g_{\text{eff}} = \frac{1}{K}\sum_{k=1}^{K} g_k, \quad B_{\text{eff}} = K \cdot B_{\text{micro}}
$$

```
  GPU can fit B_micro = 4, but we want B_eff = 32:
  K = 32/4 = 8 accumulation steps

  Step 1: Forward(batch_1) → loss_1.backward() → accumulate
  Step 2: Forward(batch_2) → loss_2.backward() → accumulate
  ...
  Step 8: Forward(batch_8) → loss_8.backward() → accumulate
  → optimizer.step()  (update using averaged gradient)
  → optimizer.zero_grad()
```

### 4. Mixed Precision Training (fp16 / bf16)

Keep a **master copy** in fp32, compute forward/backward in fp16:

```
  ┌──────────────────────────────────────────────────┐
  │             Mixed Precision Pipeline              │
  │                                                   │
  │  Master weights (fp32) ──► Cast to fp16           │
  │                               │                   │
  │                               ▼                   │
  │                    Forward pass (fp16)             │
  │                               │                   │
  │                               ▼                   │
  │                   Loss (fp32, for precision)       │
  │                               │                   │
  │                               ▼                   │
  │              Loss × scale_factor (prevent underflow)│
  │                               │                   │
  │                               ▼                   │
  │                   Backward pass (fp16 gradients)   │
  │                               │                   │
  │                               ▼                   │
  │              Gradients / scale_factor              │
  │                               │                   │
  │                               ▼                   │
  │              Update master weights (fp32)          │
  └──────────────────────────────────────────────────┘
```

**Memory savings:**

| Component | fp32 | fp16/bf16 |
|-----------|------|-----------|
| Activations | 4 bytes/param | 2 bytes/param (2× savings) |
| Gradients | 4 bytes/param | 2 bytes/param |
| Model weights | 4 bytes/param | 4 bytes (master) + 2 bytes (copy) |

**bf16 vs fp16:**

| Format | Exponent | Mantissa | Dynamic Range | Precision |
|--------|----------|----------|---------------|-----------|
| fp32 | 8 bits | 23 bits | $\pm 3.4 \times 10^{38}$ | High |
| fp16 | 5 bits | 10 bits | $\pm 6.5 \times 10^4$ | Medium |
| bf16 | 8 bits | 7 bits | $\pm 3.4 \times 10^{38}$ | Lower |

**bf16 advantage:** Same dynamic range as fp32 (no overflow/underflow), so **no loss scaling needed**.

### 5. Gradient Clipping

Prevent exploding gradients by capping the global gradient norm:

$$
\hat{g} =
\begin{cases}
g & \text{if } \lVert g \rVert \leq c \\
\frac{c}{\lVert g \rVert} \cdot g & \text{if } \lVert g \rVert > c
\end{cases}
$$

where $c$ is the max norm (typically $c = 1.0$) and:

$$
\lVert g \rVert = \sqrt{\sum_i g_i^2} \quad \text{(global norm across all parameters)}
$$

### 6. Gradient Checkpointing — Trade Compute for Memory

Instead of storing all intermediate activations for the backward pass, **recompute** them during backpropagation:

```
  Standard (store all):          Checkpointed (recompute):
  Forward:                       Forward:
  L1 → save a₁                  L1 → save a₁ (checkpoint)
  L2 → save a₂                  L2 → discard
  L3 → save a₃                  L3 → save a₃ (checkpoint)
  L4 → save a₄                  L4 → discard

  Backward:                      Backward:
  Use a₄ → grad₄                Recompute a₃→a₄, use a₄ → grad₄
  Use a₃ → grad₃                Use a₃ → grad₃
  Use a₂ → grad₂                Recompute a₁→a₂, use a₂ → grad₂
  Use a₁ → grad₁                Use a₁ → grad₁

  Memory: O(L)                   Memory: O(√L)
  Compute: 1× forward            Compute: ~1.33× forward
```

**Memory savings:** For a 32-layer model, checkpoint every 6 layers → save ~80% activation memory at the cost of ~33% extra computation.

---

## Part 2: The Universal 3-Stage Pattern

```
  Raw Base Model
       │
       ▼
  ┌──────────────────────────────┐
  │ Stage 1: SFT                 │  L = -Σₜ log P(yₜ | y₁…yₜ₋₁, I)
  │ (next-token prediction)      │  Optional: knowledge distillation
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │ Stage 2: RL (DPO / GRPO)    │  DPO: -log σ(β·Δlog π)
  │ (optimize reward signal)     │  GRPO: group-relative advantages
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │ Stage 3: Weight Merging      │  θ = θ_base + Σ λₖτₖ
  │ (combine specialists)        │  Cost: minutes, no GPU
  └──────────┬───────────────────┘
             ▼
       Final Model (deploy)
```

---

## Stage 1: Supervised Fine-Tuning (SFT)

### Cross-Entropy Loss (Selective)

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log P_\theta(y_t \mid y_1, \ldots, y_{t-1}, I)
$$

Only compute loss on **response tokens** — instruction/system tokens are masked:

```
  [system] You are helpful. [user] Describe this image. [assistant] A cat sits on a mat.
  │←── loss mask = 0 ────────────────────────────────│←── loss mask = 1 ──────────────│
```

### Knowledge Distillation (Optional)

$$
\mathcal{L}_{\text{KD}} = (1-\alpha)\mathcal{L}_{\text{CE}} + \alpha T^2 \text{KL}\!\left(\text{softmax}\!\left(\frac{z_s}{T}\right) \middle\| \text{softmax}\!\left(\frac{z_t}{T}\right)\right)
$$

```
  Teacher (large, frozen)         Student (small, training)
  ┌──────────────┐                ┌──────────────┐
  │  13B model   │ ──► z_t        │  7B model    │ ──► z_s
  └──────────────┘     │          └──────────────┘     │
                       └──────┬────────────────────────┘
                              ▼
                    KL(softmax(z_s/T) ‖ softmax(z_t/T))
```

---

## Stage 2: Reinforcement Learning — Three Algorithms

### Algorithm 1: DPO (Direct Preference Optimization)

$$
\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)
$$

**Key insight:** DPO eliminates the reward model entirely. The implicit reward is:

$$
r(x, y) = \beta \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

The partition function $Z(x)$ cancels in the preference comparison, making training stable.

### Algorithm 2: GRPO (Group Relative Policy Optimization)

For each prompt $x$, sample $G$ responses and compute group-relative advantages:

$$
A_i = \frac{R_i - \text{mean}(\{R_1, \ldots, R_G\})}{\text{std}(\{R_1, \ldots, R_G\}) + \epsilon}
$$

Update with clipped ratio:

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min\!\left(\frac{\pi_\theta(y_i)}{\pi_{\text{old}}(y_i)} A_i, \; \text{clip}\!\left(\frac{\pi_\theta(y_i)}{\pi_{\text{old}}(y_i)}, 1-\epsilon, 1+\epsilon\right) A_i\right)
$$

**GRPO advantage over PPO:** No value network needed → 50% memory savings.

### Algorithm 3: RLHF (PPO with Reward Model)

$$
\mathcal{L}_{\text{PPO}} = -\mathbb{E}\left[\min\!\left(\frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}, \; \text{clip}\!\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1 \pm \epsilon\right) \hat{A}\right)\right] + c_1 \mathcal{L}_{\text{value}} - c_2 H(\pi_\theta) + \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

### RL Algorithm Comparison

| Property | RLHF (PPO) | DPO | GRPO |
|----------|-----------|-----|------|
| Reward model | Required | Not needed | Not needed |
| Value network | Required | Not needed | Not needed |
| GPU memory | 4× model | 2× model | 2× model |
| Training stability | Low | High | High |
| Data type | Rankings | Preferences | Verifiable rewards |
| Best for | Complex rewards | Subjective quality | Math, code, factual |

---

## Stage 3: Weight-Space Merging

### Linear Interpolation (Model Soup)

$$
\theta_{\text{merged}} = \sum_{k=1}^{K} \lambda_k \theta_k, \quad \sum_k \lambda_k = 1
$$

### Task Arithmetic

$$
\theta_{\text{merged}} = \theta^{\text{base}} + \sum_k \lambda_k \underbrace{(\theta_k - \theta^{\text{base}})}_{\text{task vector } \tau_k}
$$

### TIES Merging (Trim, Elect Sign, Disjoint Merge)

1. **Trim:** Zero out parameters with smallest magnitude changes
2. **Elect Sign:** For each parameter, use the majority sign across task vectors
3. **Disjoint Merge:** Average only parameters that agree in sign

| Method | Quality | Robustness | Compute |
|--------|---------|-----------|---------|
| Linear | Good | Low | Seconds |
| Task Arithmetic | Better | Medium | Seconds |
| TIES | Best | High | Minutes |

---

## End-to-End Example 1: LLaVA-Style Pipeline

```
  Base Model (LLaMA 7B + CLIP ViT-L)
       │
       ▼
  ┌────────────────────────────────┐
  │ Stage 1a: Alignment SFT        │  558K caption pairs
  │ Train: projector only          │  1 epoch, ~4 hours (8×A100)
  └──────────┬─────────────────────┘
             ▼
  ┌────────────────────────────────┐
  │ Stage 1b: Instruction SFT      │  150K instruction data
  │ Train: projector + LLM (LoRA) │  3 epochs, ~8 hours
  └──────────┬─────────────────────┘
             ▼
  ┌────────────────────────────────┐
  │ Stage 2: DPO                   │  10K preference pairs
  │ Train: LLM (LoRA)             │  1 epoch, ~2 hours
  └──────────┬─────────────────────┘
             ▼
  ┌────────────────────────────────┐
  │ Stage 3: Merge                 │  TIES merge with DPO-free copy
  │ λ_DPO=0.7, λ_SFT=0.3         │  ~30 seconds, no GPU
  └──────────┬─────────────────────┘
             ▼
       Deploy (vLLM + LoRA serving)
```

---

## End-to-End Example 2: OCR-Specialized VLM

```
  Base Model (Qwen-VL 7B)
       │
       ▼
  ┌────────────────────────────────┐
  │ Stage 1: SFT + Distillation    │  500K OCR samples
  │ Teacher: GPT-4V OCR outputs   │  α=0.5, T=4
  │ Train: full model              │  5 epochs
  └──────────┬─────────────────────┘
             ▼
  ┌────────────────────────────────┐
  │ Stage 2: GRPO                  │  Verifiable rewards:
  │ G=8 samples per prompt        │  exact_match(OCR_pred, OCR_gt)
  │ No reward model needed!       │  2 epochs
  └──────────┬─────────────────────┘
             ▼
  ┌────────────────────────────────┐
  │ Stage 3: Model Souping         │  Greedy soup: 5 checkpoints
  │ Average best GRPO checkpoints │  Keep if val_loss improves
  └──────────┬─────────────────────┘
             ▼
       Deploy (ONNX + INT8 for edge)
```

---

## Key Equations Summary

| Concept | Formula |
|---------|---------|
| SFT | $-\sum_t \log P(y_t \mid y_{1:t-1}, I)$ |
| AdamW update | $\theta \leftarrow \theta - \eta(\hat{m}/(\sqrt{\hat{v}} + \epsilon) + \lambda\theta)$ |
| Cosine LR | $\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi \cdot \text{progress}))$ |
| Grad accumulation | $g_{\text{eff}} = \frac{1}{K}\sum_k g_k$ |
| Gradient clipping | $\hat{g} = g \cdot \min(1, c / \lVert g \rVert)$ |
| KD | $(1-\alpha)\text{CE} + \alpha T^2 \text{KL}(S \| T)$ |
| DPO | $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$ |
| GRPO advantage | $A_i = (R_i - \bar{R})/\text{std}(R)$ |
| Task Arithmetic | $\theta^{\text{base}} + \sum_k \lambda_k(\theta_k - \theta^{\text{base}})$ |

---

## Mathematical Proofs

### Proof: DPO Derivation from RLHF — Implicit Reward

**Step 1 — RLHF objective:** Maximize reward $r(x,y)$ with KL penalty to reference $\pi_{\text{ref}}$:

$$
\max_\pi \mathbb{E}_{y \sim \pi}[r(x,y)] - \beta \text{KL}(\pi(\cdot\mid x) \,\|\, \pi_{\text{ref}}(\cdot\mid x))
$$

**Step 2 — Optimal policy (closed form):**

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x,y)}{\beta}\right)
$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) e^{r(x,y)/\beta}$.

**Step 3 — Solve for implicit reward:**

$$
r(x,y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**Step 4 — Bradley-Terry preference model:** $P(y_w \succ y_l \mid x) = \sigma(r(x,y_w) - r(x,y_l))$. Substituting and canceling $Z(x)$:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta\left[\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)
$$

**∎**

#### Numerical Example

$\log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w) = 0.5$, same for loser $= -0.2$: margin $= 0.5 - (-0.2) = 0.7$. With $\beta = 0.1$: $\mathcal{L} = -\log \sigma(0.07) = -\log(0.517) = 0.659$.

---

### Proof: GRPO Group-Relative Advantage Normalization

**Step 1 — Sample $G$ responses per prompt with rewards $R_1, \ldots, R_G$.**

**Step 2 — Group statistics:**

$$
\bar{R} = \frac{1}{G}\sum_{i=1}^{G} R_i, \quad \sigma_R = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(R_i - \bar{R})^2}
$$

**Step 3 — Advantage:**

$$
A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}
$$

**Why:** Normalizing within the group removes prompt difficulty bias — only responses better than the group average get positive advantage.

**Step 4 — PPO-style clipped update:**

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right)
$$

where $\rho_i = \pi_\theta(y_i)/\pi_{\text{old}}(y_i)$. **∎**

#### Numerical Example

$G=4$, rewards $[1.0, 0.0, 0.5, 0.5]$: $\bar{R} = 0.5$, $\sigma_R = 0.408$. Advantages: $A = [1.22, -1.22, 0, 0]$. Best response gets strong positive update; worst gets negative.

---

### Proof: KL Penalty in PPO Prevents Reward Hacking

**Step 1 — Unconstrained RL:** Policy drifts to exploit spurious reward model patterns (reward hacking).

**Step 2 — KL-regularized objective:**

$$
\mathcal{L} = \mathbb{E}[\text{PPO loss}] + \beta \text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

**Step 3 — KL divergence definition:**

$$
\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right]
$$

**Step 4 — Effect:** Gradient penalizes moving too far from $\pi_{\text{ref}}$, preserving language quality while optimizing reward. As $\pi_\theta$ diverges, KL grows unboundedly — constraining the policy to the trusted region. **∎**

#### Numerical Example

If $\pi_\theta$ assigns 0.9 to a hacky response vs $\pi_{\text{ref}} = 0.01$: KL contribution $\approx \log(90) = 4.5$ nats per token — large penalty unless reward gain exceeds $\beta \times 4.5$.

---

## What You'll Build

- Complete production training loop (AdamW + cosine LR + gradient accumulation + mixed precision)
- Training visualization (loss curves, learning rate schedule)
- SFT implementation with selective loss masking
- DPO training loop from scratch
- GRPO with verifiable rewards
- Weight merging experiments (linear, task arithmetic, TIES)
- End-to-end pipeline simulation

---

## Prerequisites

- Module 03 Notebooks 01–02 (contrastive learning, pretraining objectives)
- Understanding of cross-entropy loss and optimization
- Familiarity with PyTorch training loops

---

## 🔬 Worked Examples in the Notebook

### DPO Loss — Implementation with Numerical Trace
- Step-by-step DPO computation for 4 preference pairs
- Trace: policy log-probs → reference log-probs → reward margin → sigmoid loss
- Train a simple DPO model for 100 steps
- Visualize: loss curve and reward margin (positive = policy prefers winners)

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/03_Training_Strategies/03_training_pipeline/03_training_pipeline.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| RLHF Pipeline (`../../assets/paper_figures/rlhf_pipeline.png`) | Ouyang et al. — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) | 2022 | Official HuggingFace RLHF reward-model diagram |
| RLHF Pipeline (InstructGPT) | Ouyang et al. — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) | 2022 | SFT → Reward Model → PPO |
| PPO vs DPO vs GRPO | Schulman / Rafailov / Shao | 2017-24 | 4 models vs 2 models vs 1 model comparison |

### Additional Papers Covered

- **DPO** (Rafailov et al., 2023) — Direct preference optimization, no reward model needed
- **GRPO** (Shao et al., 2024) — Group-relative advantages with verifiable rewards
- **TIES Merging** (Yadav et al., 2023) — Trim, elect sign, disjoint merge for weight merging

---

## Next Step

**[01_lora_from_scratch](../../04_Finetuning_LowCompute/01_lora_from_scratch/)** — Finetune with 100× fewer parameters using LoRA
