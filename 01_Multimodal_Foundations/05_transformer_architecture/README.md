# 05 — Transformer Architecture from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/01_Multimodal_Foundations/05_transformer_architecture/05_transformer_architecture.ipynb)

> **Time:** ~55 minutes | **Difficulty:** Advanced | **GPU Required:** No

---

## What You'll Learn

Build the complete Transformer (encoder + decoder) with every component derived:

- Pre-norm transformer blocks (MSA + FFN + residuals)
- Sinusoidal positional encoding derivation
- Causal masking for autoregressive decoding
- Full encoder-decoder for sequence-to-sequence tasks

---

## Positional Encoding — Sinusoidal Derivation

### Step 1: Motivation

Self-attention is permutation-equivariant: $\text{Attention}(\pi \mathbf{Z}) = \pi \, \text{Attention}(\mathbf{Z})$. We inject position via:

$$
\mathbf{Z}_0 = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{pos}}
$$

### Step 2: Sinusoidal Formula

For position $\text{pos}$ and dimension $i$:

$$
PE(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad
PE(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

### Step 3: Relative Position Property

Define $\omega_i = 10000^{-2i/d}$. For dimensions $(2i, 2i+1)$, position $\text{pos}$ maps to angle $\theta = \omega_i \cdot \text{pos}$ on the unit circle. Adding offset $\Delta$:

$$
PE(\text{pos}+\Delta, 2i) = \sin(\omega_i(\text{pos}+\Delta)) = \cos(\omega_i\Delta)\sin(\omega_i \cdot \text{pos}) + \sin(\omega_i\Delta)\cos(\omega_i \cdot \text{pos})
$$

This is a **rotation** by angle $\omega_i \Delta$ — relative position is encoded as a linear transform.

### Numerical Example ($d=4$, $\text{pos}=0,1,2$)

For $i=0$: $\omega_0 = 1/10000^0 = 1$.

| pos | $PE(\text{pos}, 0)=\sin(\text{pos})$ | $PE(\text{pos}, 1)=\cos(\text{pos})$ |
|-----|--------------------------------------|--------------------------------------|
| 0 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 |
| 2 | 0.909 | -0.416 |

For $i=1$: $\omega_1 = 1/10000^{2/4} = 0.01$ — slower oscillation for higher dimensions.

---

## Encoder Block — Full Derivation

### Pre-Norm Formulation (ViT / modern default)

$$
\mathbf{Z}' = \mathbf{Z} + \text{MSA}(\text{LN}(\mathbf{Z}))
$$

$$
\mathbf{Z}'' = \mathbf{Z}' + \text{FFN}(\text{LN}(\mathbf{Z}'))
$$

### FFN Expansion

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

where $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$, $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$.

**Parameter count per block:** $4d^2 + 8d^2 = 12d^2$ (MSA) + $8d^2$ (FFN) $\approx 20d^2$.

**Numerical example** ($d=128$, $L=4$ layers):

$$
\text{Params} \approx 4 \times 20 \times 128^2 = 1{,}310{,}720 \approx 1.3\text{M}
$$

---

## Causal Masking — Decoder Derivation

### Step 1: Problem

Autoregressive generation: token $t$ must not attend to tokens $> t$.

### Step 2: Mask Matrix

$$
M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}
$$

### Step 3: Masked Attention

$$
\text{CausalAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}
$$

### Numerical Example ($n=4$)

Unmasked scores (row 0): $[2.0, 1.5, 3.0, 0.5]$

With causal mask, row 0 sees only position 0:

$$
\text{softmax}([2.0, -\infty, -\infty, -\infty]) = [1.0, 0, 0, 0]
$$

Row 2 sees positions 0, 1, 2:

$$
\text{softmax}([1.0, 0.5, 2.0, -\infty]) = \left[\frac{e^1}{e^1+e^{0.5}+e^2}, \frac{e^{0.5}}{Z}, \frac{e^2}{Z}, 0\right] \approx [0.114, 0.069, 0.817, 0]
$$

---

## Encoder-Decoder Cross-Attention

Decoder queries attend to encoder keys/values:

$$
\mathbf{Q} = \mathbf{Z}_{\text{dec}} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Z}_{\text{enc}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{Z}_{\text{enc}} \mathbf{W}_V
$$

Used in captioning (Module 02): image encoder output → decoder cross-attention.

---

## Full Pipeline — Tensor Shapes

```
Input tokens (B, T) → Embed → (B, T, d) → + PE → (B, T, d)
    → Encoder × L → (B, T, d)
    → Decoder (causal self-attn + cross-attn + FFN) × L → (B, T, d)
    → Linear head → (B, T, vocab_size)
```

**Numerical example:** $B=2$, $T=8$, $d=64$, $L=2$, vocab=1000:

- Encoder output: $(2, 8, 64)$
- Logits: $(2, 8, 1000)$
- Cross-entropy over 8000 token predictions per batch

---

## What You'll Build

- `TransformerEncoderLayer` and `TransformerDecoderLayer`
- Sinusoidal + learned positional encoding
- Mini seq2seq model (~200K params) on synthetic copy task

---

## Next Step

**[06_tokenization_embeddings/06_tokenization_embeddings.ipynb](../06_tokenization_embeddings/06_tokenization_embeddings.ipynb)**
