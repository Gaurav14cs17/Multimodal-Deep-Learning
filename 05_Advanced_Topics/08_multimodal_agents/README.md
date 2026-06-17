# 08 — Multimodal Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/05_Advanced_Topics/08_multimodal_agents/08_multimodal_agents.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

Agents that perceive, reason, and act with multimodal input:

- Tool use with visual grounding
- ReAct-style planning loops
- Perceive → reason → act architecture
- Build a minimal multimodal agent

---

## Agent Loop — Mathematical Formulation

### Step 1: State

At step $t$, agent state $s_t$ includes conversation history, visual observations, and tool outputs:

$$
s_t = (I, Q, \{(a_1, o_1), \ldots, (a_{t-1}, o_{t-1})\})
$$

where $a_i$ is action (tool call or text) and $o_i$ is observation.

### Step 2: Policy

Multimodal LLM defines policy:

$$
a_t \sim \pi_\theta(a \mid s_t, I)
$$

### Step 3: Environment Update

$$
s_{t+1} = \text{Update}(s_t, a_t, o_t), \quad o_t = \text{Env}(a_t)
$$

### Step 4: Termination

Stop when $a_t = \text{ANSWER}$ or $t \geq T_{\max}$.

---

## ReAct — Reasoning + Acting Derivation

Interleave thought, action, observation:

$$
\tau = (r_1, a_1, o_1, r_2, a_2, o_2, \ldots, r_T, a_T)
$$

Joint probability under autoregressive policy:

$$
P(\tau \mid I, Q) = \prod_{t=1}^{T} P(r_t \mid s_t) \cdot P(a_t \mid s_t, r_t) \cdot \mathbb{1}[o_t = \text{Env}(a_t)]
$$

**Training:** Supervised fine-tuning on trajectories with human or GPT-4 annotations.

### Numerical Example — 3-Step Trajectory

Task: "How many cars in the image?"

| Step | Type | Content | $P(\cdot)$ |
|------|------|---------|-----------|
| 1 | Thought $r_1$ | "I need to detect cars" | 0.88 |
| 1 | Action $a_1$ | `detect_objects(class="car")` | 0.91 |
| 1 | Obs $o_1$ | `[bbox1, bbox2, bbox3]` | 1.0 (deterministic) |
| 2 | Thought $r_2$ | "I count 3 bounding boxes" | 0.85 |
| 2 | Action $a_2$ | `ANSWER(3)` | 0.93 |

Trajectory probability: $0.88 \times 0.91 \times 0.85 \times 0.93 \approx 0.63$

---

## Visual Tool Use — Grounding Derivation

Tool call specifies region of interest:

$$
\text{Tool}(\text{"crop"}, \mathbf{b}) \rightarrow I_{\text{crop}} = \text{Crop}(I, \mathbf{b})
$$

Follow-up VQA on crop:

$$
a = \text{VLM}(I_{\text{crop}}, Q_{\text{followup}})
$$

**Numerical example:** Full image $224 \times 224$, crop bbox $(0.4, 0.3, 0.2, 0.2)$ in normalized coords:

- Pixel crop: $x \in [89, 134]$, $y \in [67, 112]$
- Cropped region: $45 \times 45$ pixels → zoomed view of object

---

## Planning with Visual Memory

Maintain visual memory bank $\mathcal{M} = \{(I_i, \mathbf{h}_i, t_i)\}$:

$$
\text{Retrieve}(Q) = \arg\max_{i} \cos(\text{Enc}(Q), \mathbf{h}_i)
$$

Agent retrieves relevant past frames for video understanding or multi-image tasks.

**Numerical example:** 3 stored frames with embeddings, query "What color was the car in frame 2?":

| Frame | $\cos(q, \mathbf{h}_i)$ |
|-------|------------------------|
| 1 | 0.32 |
| 2 | 0.89 |
| 3 | 0.41 |

Retrieve frame 2 → answer from that visual context.

---

## Agent Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                  Multimodal Agent                        │
  │                                                          │
  │  Input: Image I + Question Q                             │
  │       │                                                  │
  │       ▼                                                  │
  │  ┌──────────────┐                                        │
  │  │ Vision Encoder│ → visual tokens                       │
  │  └──────┬───────┘                                        │
  │         │                                                │
  │         ▼                                                │
  │  ┌──────────────┐     ┌─────────────┐                     │
  │  │ Multimodal   │◄───►│ Tool Router │                     │
  │  │ LLM (policy) │     │ detect/crop │                     │
  │  └──────┬───────┘     │ calc/search │                     │
  │         │             └──────┬──────┘                     │
  │         │                    │ observations              │
  │         ▼                    │                            │
  │  Thought → Action → Observation → ... → ANSWER           │
  └─────────────────────────────────────────────────────────┘
```

---

## Mathematical Proofs

### Proof: Planning Formalization — MDP View of Agent Loop

**Step 1 — Define MDP:** $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$ where state $s_t$ includes $(I, Q, \text{history})$.

**Step 2 — State transition:**

$$
s_{t+1} = \text{Update}(s_t, a_t, o_t), \quad o_t = \text{Env}(a_t)
$$

**Step 3 — Policy:** Multimodal LLM $\pi_\theta(a_t \mid s_t, I)$ selects tool call or answer.

**Step 4 — Objective:** Maximize expected cumulative reward:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]
$$

For QA tasks, $r = 1$ if final answer correct, 0 otherwise — sparse terminal reward. **∎**

#### Numerical Example

3-step trajectory with rewards $[0, 0, 1]$, $\gamma = 0.99$: return $= 0 + 0 + 0.99^2 \times 1 = 0.98$.

---

### Proof: ReAct Trajectory Probability

**Step 1 — Trajectory:** $\tau = (r_1, a_1, o_1, \ldots, r_T, a_T)$.

**Step 2 — Factorization:**

$$
P(\tau \mid I, Q) = \prod_{t=1}^{T} P(r_t \mid s_t) \cdot P(a_t \mid s_t, r_t) \cdot \mathbb{1}[o_t = \text{Env}(a_t)]
$$

**Step 3 — Observations are deterministic** given actions — $\mathbb{1}[\cdot]$ selects consistent env transitions.

**Step 4 — Training:** Maximize $\log P(\tau^* \mid I, Q)$ on expert trajectories (SFT) or use RL for reward optimization. **∎**

#### Numerical Example

From table: $P(\tau) = 0.88 \times 0.91 \times 1.0 \times 0.85 \times 0.93 = 0.63$ for successful 2-step car-counting task.

---

## What You'll Build

- Minimal ReAct loop with mock tools (`detect`, `count`, `answer`)
- Visual crop tool on synthetic image grid
- Trajectory logger with step probabilities

---

## Next Step

Return to **[../../README.md](../../README.md)** for the full course overview.
