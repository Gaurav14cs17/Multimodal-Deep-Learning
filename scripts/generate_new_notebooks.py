#!/usr/bin/env python3
"""Generate substantial content for new/skeleton notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def colab_cell(module_path: str) -> str:
    return f'''# ============================================================
#  Google Colab Setup — Run this cell FIRST
# ============================================================
import os, sys

try:
    import google.colab
    IN_COLAB = True
    print("Google Colab detected — setting up environment...")
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_URL = "https://github.com/Gaurav14cs17/Multimodal-Deep-Learning.git"
    REPO_DIR = "/content/Multimodal-Deep-Learning"

    if not os.path.exists(REPO_DIR):
        print("Cloning repository...")
        !git clone --depth 1 {{REPO_URL}} {{REPO_DIR}}
    else:
        print("Repository already cloned")

    print("Installing dependencies...")
    !pip install -q -r {{REPO_DIR}}/requirements.txt

    MODULE_DIR = f"{{REPO_DIR}}/{module_path}"
    os.chdir(MODULE_DIR)
    os.makedirs(f"{{REPO_DIR}}/assets", exist_ok=True)

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    print(f"Colab setup complete — {{os.getcwd()}}")

    import torch
    if torch.cuda.is_available():
        print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    else:
        print("Device: CPU (all notebooks work fine on CPU)")
else:
    os.makedirs("../../assets", exist_ok=True)
    print("Running locally — all set!")'''


def imports_cell() -> str:
    return '''import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

try:
    from utils.visualization import set_style
    from utils.helpers import count_parameters, get_device
    set_style()
except ImportError:
    def set_style():
        plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 100})
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total:,}")
        return total
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_style()

torch.manual_seed(42)
np.random.seed(42)
device = get_device() if callable(get_device) else torch.device('cpu')
print(f"PyTorch {torch.__version__} | Device: {device}")'''


def nb(cells, metadata=None):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": metadata or {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def write_nb(rel_path, cells):
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb(cells), indent=1))
    print(f"Wrote {rel_path} ({sum(1 for c in cells if c['cell_type']=='code')} code cells)")


def standard_header(title, bullets, colab_url_suffix):
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/{colab_url_suffix})"
    body = "\n".join(f"- {b}" for b in bullets)
    return md(f"""{badge}

# {title}

**This notebook covers:**
{body}

**Runtime:** ~10–15 minutes on CPU

---

> **Theory & derivations:** See [README.md](./README.md) for full step-by-step math.
""")


NOTEBOOKS = {}

# --- 04 Attention ---
NOTEBOOKS["01_Multimodal_Foundations/04_attention_mechanism/04_attention_mechanism.ipynb"] = {
    "module": "01_Multimodal_Foundations/04_attention_mechanism",
    "header": ("04. Attention Mechanism Deep Dive", [
        "Scaled dot-product self-attention from first principles",
        "Multi-head attention implementation",
        "Attention heatmap visualization",
        "Cross-attention for multimodal fusion",
        "Numerical step-by-step trace with real numbers",
    ], "01_Multimodal_Foundations/04_attention_mechanism/04_attention_mechanism.ipynb"),
    "cells": [
        md("""## 1. Scaled Dot-Product Attention

$$\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V$$

Division by $\\sqrt{d_k}$ keeps dot-product variance at 1 as $d_k$ grows.
"""),
        code('''def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Numerical trace: 3 tokens, d_k=4 (from README worked example)
Q = torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]]])
K = torch.tensor([[[1., 1., 0., 0.], [0., 0., 1., 1.], [1., 0., 1., 0.]]])
V = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])

raw = Q @ K.transpose(-2, -1)
scaled = raw / (4 ** 0.5)
print("QK^T / sqrt(d_k) =\\n", scaled[0].numpy().round(3))
out, attn = scaled_dot_product_attention(Q, K, V)
print("\\nAttention weights (row 0):", attn[0, 0].numpy().round(3), " sum=", attn[0, 0].sum().item())
print("Output token 0:", out[0, 0].numpy().round(3))'''),
        md("""## 2. Multi-Head Attention

Each head learns a different relational pattern; outputs are concatenated and projected.
"""),
        code('''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output), attn_weights

mha = MultiHeadAttention(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)
out, w = mha(x, x, x)
print(f"Input {x.shape} -> Output {out.shape}, attn {w.shape}")
count_parameters(mha)'''),
        md("""## 3. Attention Heatmap Visualization
"""),
        code('''tokens = ['CLS', 'a', 'cat', 'sits']
x = torch.randn(1, len(tokens), 64)
with torch.no_grad():
    _, attn = mha(x, x, x)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for h, ax in enumerate(axes):
    im = ax.imshow(attn[0, h].numpy(), cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)), labels=tokens)
    ax.set_yticks(range(len(tokens)), labels=tokens)
    ax.set_title(f'Head {h+1}')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('Multi-Head Self-Attention Weights')
plt.tight_layout()
plt.show()'''),
        md("""## 4. Cross-Attention (Multimodal Fusion)

Text tokens **query** image patch keys/values: $\\text{CrossAttn}(Z_{txt}, Z_{img})$.
"""),
        code('''class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, query_seq, kv_seq):
        return self.mha(query_seq, kv_seq, kv_seq)

cross = CrossAttention(64, num_heads=4)
text = torch.randn(1, 4, 64)   # 4 text tokens
image = torch.randn(1, 16, 64)  # 16 patch tokens
fused, cross_attn = cross(text, image)
print(f"Text {text.shape} attends to image {image.shape} -> {fused.shape}")
print(f"Cross-attn matrix shape: {cross_attn.shape}  (heads, text, patches)")'''),
        md("""## 5. Causal Masking Preview

Decoder self-attention masks future positions with $M_{ij} = -\\infty$ when $i < j$.
"""),
        code('''seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
x = torch.randn(1, seq_len, 64)
with torch.no_grad():
    _, masked_attn = mha(x, x, x, mask=causal_mask.unsqueeze(0).unsqueeze(0))
print("Causal attention (upper triangle should be ~0):")
print(masked_attn[0, 0].numpy().round(3))'''),
        md("""## 6. Complexity Analysis

Self-attention over $n$ tokens costs $O(n^2 d)$ per layer. Cross-attention between $T$ text and $N$ image tokens costs $O(T \\cdot N \\cdot d)$.
"""),
        code('''def attention_flops(n_tokens, d_model, n_heads):
    d_k = d_model // n_heads
    qkv = 3 * n_tokens * d_model * d_model
    scores = n_tokens * n_tokens * d_k * n_heads
    weighted = n_tokens * n_tokens * d_k * n_heads
    out_proj = n_tokens * d_model * d_model
    return qkv + scores + weighted + out_proj

for n in [64, 128, 256]:
    flops = attention_flops(n, 256, 4)
    print(f"n={n:3d} tokens -> ~{flops/1e6:.2f}M multiply-adds per layer")'''),
        md("""## Summary

- Built scaled dot-product and multi-head attention from scratch
- Visualized attention heatmaps and cross-modal fusion
- Traced a numerical example matching the README derivation

**Next:** [05_transformer_architecture](../05_transformer_architecture/05_transformer_architecture.ipynb)
"""),
    ],
}

# --- 05 Transformer ---
NOTEBOOKS["01_Multimodal_Foundations/05_transformer_architecture/05_transformer_architecture.ipynb"] = {
    "module": "01_Multimodal_Foundations/05_transformer_architecture",
    "header": ("05. Transformer Architecture", [
        "Sinusoidal positional encoding",
        "Encoder block (LayerNorm + MHA + FFN + residuals)",
        "Full Transformer Encoder",
        "Decoder block with causal masking",
        "Autoregressive inference example",
    ], "01_Multimodal_Foundations/05_transformer_architecture/05_transformer_architecture.ipynb"),
    "cells": [
        md("""## 1. Sinusoidal Positional Encoding

$$PE(pos, 2i) = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE(pos, 2i+1) = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)$$
"""),
        code('''def sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = sinusoidal_pe(50, 64)
plt.figure(figsize=(10, 4))
plt.imshow(pe.T.numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
plt.xlabel('Position'); plt.ylabel('Dimension'); plt.title('Sinusoidal Positional Encoding')
plt.colorbar(label='value')
plt.tight_layout(); plt.show()'''),
        md("""## 2. Encoder Block (Pre-Norm)

$$Z' = \\text{MSA}(\\text{LN}(Z)) + Z, \\quad Z'' = \\text{FFN}(\\text{LN}(Z')) + Z'$$
"""),
        code('''class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

# Reuse MHA from attention notebook (inline for standalone run)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        def proj(x, W):
            return W(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        Qh, Kh, Vh = proj(Q, self.W_q), proj(K, self.W_k), proj(V, self.W_v)
        scores = (Qh @ Kh.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ Vh).transpose(1, 2).contiguous().view(B, -1, Q.size(-1))
        return self.W_o(out), attn

block = EncoderBlock(128, 4)
x = torch.randn(2, 16, 128)
y = block(x)
print(f"Encoder block: {x.shape} -> {y.shape}")
count_parameters(block)'''),
        md("""## 3. Full Transformer Encoder
"""),
        code('''class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Parameter(sinusoidal_pe(max_len, d_model), requires_grad=False)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids, mask=None):
        x = self.embed(token_ids) + self.pe[:token_ids.size(1)]
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln(x)

enc = TransformerEncoder(vocab_size=100, d_model=128, n_layers=3)
ids = torch.randint(0, 100, (2, 20))
h = enc(ids)
print(f"Token ids {ids.shape} -> encoder output {h.shape}")
count_parameters(enc)'''),
        md("""## 4. Decoder Block with Causal Mask
"""),
        code('''class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, causal_mask=None):
        sa, _ = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), causal_mask)
        x = x + self.dropout(sa)
        ca, _ = self.cross_attn(self.ln2(x), enc_out, enc_out)
        x = x + self.dropout(ca)
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x

dec_block = DecoderBlock(128, 4)
T = 8
causal = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
dec_in = torch.randn(1, T, 128)
enc_out = torch.randn(1, 16, 128)
dec_out = dec_block(dec_in, enc_out, causal)
print(f"Decoder: {dec_in.shape} + encoder {enc_out.shape} -> {dec_out.shape}")'''),
        md("""## 5. Autoregressive Inference (Greedy Decoding)
"""),
        code('''class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size=50, d_model=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Parameter(sinusoidal_pe(64, d_model), requires_grad=False)
        self.layers = nn.ModuleList([DecoderBlock(d_model, 4) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids, enc_out):
        x = self.embed(ids) + self.pe[:ids.size(1)]
        T = ids.size(1)
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, enc_out, mask)
        return self.head(self.ln(x))

lm = TinyDecoderLM()
enc_out = torch.randn(1, 8, 64)
generated = [5]
for _ in range(6):
    ids = torch.tensor([generated])
    logits = lm(ids, enc_out)[:, -1]
    next_tok = logits.argmax(dim=-1).item()
    generated.append(next_tok)
print("Greedy generation (token ids):", generated)'''),
        md("""## Summary

Built positional encoding, encoder/decoder blocks, and a tiny autoregressive decoder.

**Next:** [06_tokenization_embeddings](../06_tokenization_embeddings/06_tokenization_embeddings.ipynb)
"""),
    ],
}

# Import extended specs
import sys as _sys
_sys.path.insert(0, str(ROOT / "scripts"))
from notebook_specs_extended import EXTENDED_NOTEBOOKS

NOTEBOOKS.update(EXTENDED_NOTEBOOKS)

if __name__ == "__main__":
    for rel, spec in NOTEBOOKS.items():
        title, bullets, url = spec["header"]
        cells = [standard_header(title, bullets, url), code(colab_cell(spec["module"])), code(imports_cell())]
        cells.extend(spec["cells"])
        write_nb(rel, cells)
    print(f"Generated {len(NOTEBOOKS)} notebooks.")
