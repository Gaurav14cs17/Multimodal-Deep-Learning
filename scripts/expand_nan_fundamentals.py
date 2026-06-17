#!/usr/bin/env python3
"""Insert IEEE 754 fundamentals, 15 NaN sources, debugging methodology into Module 07 notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB05 = ROOT / "07_traing_problem_and soultion/05_training_instability/05_training_instability.ipynb"
NB06 = ROOT / "07_traing_problem_and soultion/06_distributed_training/06_distributed_training.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Part 1–3: Insert after imports (index 3) in notebook 05
# ---------------------------------------------------------------------------
IEEE_AND_SOURCES = [
    md("""## 0. How NaN is Created — IEEE 754 Arithmetic Rules

Every floating-point NaN in PyTorch comes from one of **seven IEEE 754 operations** (or propagation from an existing NaN):

| # | Operation | Result |
|---|-----------|--------|
| 1 | $0 / 0$ | NaN |
| 2 | $\\infty - \\infty$ | NaN |
| 3 | $\\infty + (-\\infty)$ | NaN |
| 4 | $0 \\times \\infty$ | NaN |
| 5 | $\\infty / \\infty$ | NaN |
| 6 | $\\sqrt{\\text{negative}}$ | NaN |
| 7 | Any op involving existing NaN | NaN (propagation) |

### Bit-Level Representation

**FP32:** sign (1 bit) + exponent (8 bits) + mantissa (23 bits)

- **NaN:** all exponent bits = 1, mantissa $\\neq 0$
- **Inf:** all exponent bits = 1, mantissa = 0
- **Normal:** exponent $\\in [1, 254]$

**FP16:** sign (1) + exponent (5) + mantissa (10) → **max = 65504**

**BF16:** sign (1) + exponent (8) + mantissa (7) → **max $\\approx 3.39 \\times 10^{38}$** (same exponent range as FP32!)

This is why BF16 rarely overflows where FP16 dies: same dynamic range, fewer mantissa bits.
"""),

    code("""# Demonstrate all 7 IEEE 754 NaN creation rules
rules = [
    ("0/0", lambda: torch.tensor(0.) / 0),
    ("inf - inf", lambda: torch.tensor(float('inf')) - torch.tensor(float('inf'))),
    ("inf + (-inf)", lambda: torch.tensor(float('inf')) + torch.tensor(float('-inf'))),
    ("0 * inf", lambda: torch.tensor(0.) * torch.tensor(float('inf'))),
    ("inf / inf", lambda: torch.tensor(float('inf')) / torch.tensor(float('inf'))),
    ("sqrt(-1)", lambda: torch.sqrt(torch.tensor(-1.))),
    ("NaN propagation", lambda: torch.tensor(float('nan')) + 1),
]
print("IEEE 754 NaN creation rules:")
for name, fn in rules:
    result = fn().item()
    print(f"  {name:20s} = {result}")
"""),

    code("""# FP16 vs BF16 vs FP32 — range, epsilon, subnormals
dtypes = [torch.float32, torch.float16, torch.bfloat16]
rows = []
for dt in dtypes:
    fi = torch.finfo(dt)
    rows.append({
        'dtype': str(dt).split('.')[-1],
        'max': fi.max,
        'min': fi.min,
        'smallest_normal': fi.tiny,
        'eps': fi.eps,
        'bits': {torch.float32: '1+8+23', torch.float16: '1+5+10', torch.bfloat16: '1+8+7'}[dt],
    })

print(f"{'dtype':<10} {'bits':<8} {'max':>12} {'min_normal':>12} {'eps':>12}")
print("-" * 58)
for r in rows:
    print(f"{r['dtype']:<10} {r['bits']:<8} {r['max']:>12.4e} {r['smallest_normal']:>12.4e} {r['eps']:>12.4e}")

# Visual: representable range comparison
fig, ax = plt.subplots(figsize=(10, 4))
labels = [r['dtype'] for r in rows]
maxs = [r['max'] for r in rows]
colors = ['#2196F3', '#FF5722', '#4CAF50']
bars = ax.bar(labels, maxs, color=colors)
ax.set_yscale('log')
ax.set_ylabel('Max representable value')
ax.set_title('FP32 vs FP16 vs BF16 — Dynamic Range (log scale)')
for bar, val in zip(bars, maxs):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.2, f'{val:.2e}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

print("\\nKey insight: FP16 max=65504 — any activation/logit above this overflows to Inf → NaN.")
print("BF16 shares FP32 exponent range — preferred for transformer/contrastive training on A100/H100.")
"""),

    md("""## 0b. NaN is Contagious — The Propagation Problem

Once NaN enters **any** tensor, it spreads irreversibly:

```
Forward pass:  NaN input → NaN output (through EVERY subsequent layer)
Backward pass: NaN gradient → NaN weight update → ALL future outputs NaN
Distributed:   NaN on ONE GPU → AllReduce average → NaN on ALL GPUs
```

**The ONLY way to recover:** detect NaN **before** it propagates, then skip the step or fix the source.

There is no "NaN recovery" mid-training — you must restart from the last good checkpoint after fixing the root cause.
"""),

    md("""## 0c. `torch.autograd.detect_anomaly()` — The #1 Debugging Tool

**THE FIRST THING TO TRY when loss goes NaN.**

What it does:
- Wraps every backward op with a NaN/Inf check
- Prints a **full traceback** showing exactly which operation produced NaN
- Performance cost: ~**2× slower** (use only for debugging, not production)

How to read the error:
```
RuntimeError: Function 'DivBackward0' returned nan values in its 0th output.
```
→ Division produced NaN — check for division by zero in forward pass.

Persistent mode: `torch.autograd.set_detect_anomaly(True)` for entire training loop (very slow).
"""),

    code("""# detect_anomaly demo — pinpoints EXACT op that produces NaN
class NaNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.lin.weight.data.fill_(0.0)  # zero weights → zero output

    def forward(self, x):
        out = self.lin(x)  # all zeros when weights are zero
        # Bad: divide by norm without epsilon → 0/0 = NaN in backward
        return out / out.norm(dim=-1, keepdim=True)

model = NaNModel()
x = torch.randn(2, 4, requires_grad=True)
target = torch.randn(2, 4)

print("Without detect_anomaly (NaN silently propagates):")
out = model(x)
loss = (out - target).pow(2).mean()
loss.backward()
print(f"  loss={loss.item()}, grad has NaN: {torch.isnan(x.grad).any().item()}")

print("\\nWith detect_anomaly (shows exact backward op):")
model.zero_grad()
x.grad = None
try:
    with torch.autograd.detect_anomaly():
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
except RuntimeError as e:
    print(f"  Caught: {str(e)[:120]}...")

# Persistent mode (comment out in production — very slow)
# torch.autograd.set_detect_anomaly(True)
print("\\nTip: combine detect_anomaly with forward hooks (NaNTracer below) for full picture.")
"""),

    md("""---

## 0d. Every Common NaN Source in Deep Learning (15 Cases)

Each subsection: **WHY** (math) → **HOW to detect** → **HOW to fix** → runnable demo.

Use this as a lookup table when `detect_anomaly()` points to a suspicious op.
"""),

    md("""### Source 1: `log(0)` → $-\\infty$ → NaN in Loss

**WHY:** Cross-entropy internally computes $\\log(\\text{softmax}(x))$. When softmax output $\\approx 0$ (confident wrong prediction): $\\log(0) = -\\infty$. Combining $0 \\times (-\\infty) = \\text{NaN}$ in weighted losses.

**Detect:** NaN in loss backward; `detect_anomaly` points to `LogBackward`.

**Fix:** Use `F.log_softmax` + `F.nll_loss`, or clamp probs with epsilon.
"""),

    code("""# Source 1: log(0) in cross-entropy path
probs = torch.tensor([0.0, 1.0])
raw_log = -torch.log(probs[0])  # inf
print(f"log(0): {raw_log.item()}  (inf → NaN when multiplied by 0*inf elsewhere)")

# Fix 1: epsilon clamp
safe_log = torch.log(probs + 1e-8)
print(f"log(prob + eps): {safe_log[0].item():.4f}")

# Fix 2: log_softmax (numerically stable — PyTorch uses log-sum-exp internally)
logits = torch.tensor([10.0, -5.0])  # confident wrong class
stable = F.log_softmax(logits, dim=0)
loss = F.nll_loss(stable.unsqueeze(0), torch.tensor([0]))
print(f"log_softmax + nll_loss: {loss.item():.4f} (finite)")
"""),

    md("""### Source 2: KL Divergence NaN

**WHY:** $\\text{KL}(p \\| q) = \\sum_i p_i \\log(p_i / q_i)$. When $q_i \\approx 0$: $\\log(p/q) \\to \\infty$. When $p_i = 0$: $0 \\cdot \\log(0) = 0 \\cdot (-\\infty) = \\text{NaN}$ (IEEE 754 rule #4 variant).

**Detect:** NaN in distillation/VAE losses.

**Fix:** `reduction='batchmean'`, clamp $q \\geq \\epsilon$, use `F.kl_div` with log-prob input.
"""),

    code("""# Source 2: KL divergence NaN
p = torch.tensor([0.5, 0.5, 0.0])
q = torch.tensor([0.3, 0.3, 0.0001])

# Bad: q nearly zero on third class, p=0 on third → NaN
kl_bad = (p * (p / q).log()).sum()
print(f"Naive KL: {kl_bad.item()}  (NaN={torch.isnan(kl_bad).item()})")

# Fix: clamp q, use F.kl_div
q_safe = q.clamp(min=1e-8)
kl_good = F.kl_div(q_safe.log(), p, reduction='batchmean')
print(f"F.kl_div (clamped): {kl_good.item():.4f}")
"""),

    md("""### Source 3: Cosine Similarity with Zero Vectors

**WHY:** $\\cos(a, b) = \\frac{a \\cdot b}{\\|a\\| \\|b\\|}$. When $\\|a\\| = 0$: division by zero → NaN. Common in contrastive learning when an encoder outputs all zeros (dead ReLU, collapsed representation).

**Detect:** NaN in CLIP/SimCLR similarity matrix.

**Fix:** Pass `eps=1e-8` to `F.cosine_similarity`, or check for zero-norm vectors before normalization.
"""),

    code("""# Source 3: cosine similarity with zero vector
a = torch.zeros(5)
b = torch.randn(5)
cos_bad = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
print(f"Zero vector cosine: {cos_bad.item()}  (NaN={torch.isnan(cos_bad).item()})")

cos_safe = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), eps=1e-8)
print(f"With eps=1e-8: {cos_safe.item()}")
"""),

    md("""### Source 4: Layer Normalization with Zero Variance

**WHY:** $\\text{LN}(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}$. When all elements equal: $\\sigma^2 = 0$, relies entirely on $\\epsilon$. With tiny $\\epsilon$ (e.g. $10^{-12}$) and **FP16**: $\\sqrt{\\epsilon}$ may underflow → division by zero.

**Detect:** NaN in first LN layer with constant input; common in FP16 training.

**Fix:** Use $\\epsilon \\geq 10^{-6}$ for FP16, $\\epsilon \\geq 10^{-5}$ as safe default.
"""),

    code("""# Source 4: LayerNorm zero variance + tiny eps in FP16
x = torch.ones(1, 10) * 5.0  # zero variance

ln_bad = nn.LayerNorm(10, eps=1e-12)
out_bad = ln_bad(x.half())
print(f"LN eps=1e-12, FP16: NaN={torch.isnan(out_bad).any().item()}")

ln_good = nn.LayerNorm(10, eps=1e-5)
out_good = ln_good(x.half())
print(f"LN eps=1e-5, FP16: NaN={torch.isnan(out_good).any().item()}, mean={out_good.mean().item():.4f}")
"""),

    md("""### Source 5: Cross-Entropy with Out-of-Range Labels

**WHY:** Labels must satisfy $0 \\leq y_i < C$. Label $y=12$ with $C=10$ classes → undefined indexing in CUDA kernel → garbage or NaN.

**Detect:** NaN on first batch; check `labels.min()`, `labels.max()` vs `logits.size(-1)`.

**Fix:** Validate labels before loss; fix data pipeline.
"""),

    code("""# Source 5: out-of-range labels
logits = torch.randn(4, 10)
labels = torch.tensor([0, 5, 12, -1])  # 12 and -1 invalid!

print(f"Label range: [{labels.min()}, {labels.max()}], num_classes={logits.size(-1)}")
try:
    loss = F.cross_entropy(logits, labels)
    print(f"Loss: {loss.item()}  NaN={torch.isnan(loss).item()}")
except Exception as e:
    print(f"Error: {e}")

# Fix: validate before loss
valid = labels.min() >= 0 and labels.max() < logits.size(-1)
print(f"Labels valid: {valid.item()}")
labels_fixed = labels.clamp(0, logits.size(-1) - 1)
loss_fixed = F.cross_entropy(logits, labels_fixed)
print(f"Fixed loss: {loss_fixed.item():.4f}")
"""),

    md("""### Source 6: Empty Batch After Filtering → NaN

**WHY:** After filtering bad samples, batch size may become 0. `tensor.mean()` on empty tensor returns NaN (PyTorch convention: undefined reduction).

**Detect:** NaN loss intermittently; check if DataLoader returns empty batches.

**Fix:** Skip empty batches: `if batch.size(0) == 0: continue`
"""),

    code("""# Source 6: empty batch
batch = torch.randn(0, 10)
print(f"Empty batch shape: {batch.shape}")
mean_bad = batch.mean()
print(f"mean(empty): {mean_bad.item()}  (NaN={torch.isnan(mean_bad).item()})")

# Fix
if batch.size(0) > 0:
    mean_good = batch.mean()
    print(f"mean(non-empty): {mean_good.item()}")
else:
    print("Skip: empty batch detected before forward pass")
"""),

    md("""### Source 7: Embedding with Out-of-Vocabulary Index

**WHY:** `nn.Embedding(vocab_size, dim)` expects indices $\\in [0, \\text{vocab\\_size})$. Index $\\geq \\text{vocab\\_size}$ → out-of-bounds memory access → garbage or NaN.

**Detect:** NaN in embedding output; check max token ID vs vocab size.

**Fix:** Clamp indices: `ids.clamp(0, emb.num_embeddings - 1)` or use `<unk>` token.
"""),

    code("""# Source 7: OOV embedding index
emb = nn.Embedding(1000, 64)
ids = torch.tensor([500, 1500])  # 1500 >= 1000!
print(f"Max id={ids.max().item()}, vocab_size={emb.num_embeddings}")

try:
    out_bad = emb(ids)
    print(f"OOV embedding output has NaN: {torch.isnan(out_bad).any().item()}")
except IndexError as e:
    print(f"OOV index crash (expected): {e}")

ids_safe = ids.clamp(0, emb.num_embeddings - 1)
out_safe = emb(ids_safe)
print(f"Clamped ids: {ids_safe.tolist()}, NaN={torch.isnan(out_safe).any().item()}")
"""),

    md("""### Source 8: Division in Custom Loss Functions (Focal Loss)

**WHY:** Focal loss: $(1-p)^\\gamma \\cdot (-\\log p)$. When $p=0$ and bad implementation: $\\log(0)=-\\infty$, weight $=1$ → $\\infty$. When $p=1$ and wrong class: $0 \\cdot (-\\log 0) = 0 \\cdot \\infty = \\text{NaN}$.

**Detect:** NaN in object detection / imbalanced classification.

**Fix:** Clamp predictions: `pred.clamp(1e-7, 1 - 1e-7)`.
"""),

    code("""# Source 8: focal loss NaN
def focal_loss_bad(pred, target, gamma=2):
    ce = -target * torch.log(pred)  # log(0) = -inf when pred=0
    weight = (1 - pred) ** gamma
    return (weight * ce).mean()

def focal_loss_safe(pred, target, gamma=2):
    pred = pred.clamp(1e-7, 1 - 1e-7)
    ce = -target * torch.log(pred)
    weight = (1 - pred) ** gamma
    return (weight * ce).mean()

pred = torch.tensor([0.0, 0.9, 1.0])
target = torch.tensor([1.0, 0.0, 1.0])  # pred=0, target=1 → log(0)

loss_bad = focal_loss_bad(pred, target)
print(f"Bad focal loss: {loss_bad.item()}  (NaN={torch.isnan(loss_bad).item()})")
loss_safe = focal_loss_safe(pred, target)
print(f"Safe focal loss: {loss_safe.item():.4f}")
"""),

    md("""### Source 9: NaN from Large Weight Initialization

**WHY:** Weights initialized with $\\text{std} \\gg 1$ → activations grow as $\\text{std}^{L}$ through $L$ layers → exceed FP32 max ($\\approx 3.4 \\times 10^{38}$) → Inf → NaN in first forward pass.

**Detect:** NaN at step 0; check `out.abs().max()` after first forward.

**Fix:** Kaiming/Xavier init; for transformers use $\\mathcal{N}(0, 0.02)$ or $\\mu$P scaling.
"""),

    code("""# Source 9: large weight initialization
model_bad = nn.Linear(1000, 1000)
nn.init.normal_(model_bad.weight, std=10.0)  # way too large
x = torch.randn(32, 1000)
out_bad = model_bad(x)
print(f"Bad init — max activation: {out_bad.abs().max().item():.2e}, NaN={torch.isnan(out_bad).any().item()}")

model_good = nn.Linear(1000, 1000)
nn.init.kaiming_normal_(model_good.weight, mode='fan_in', nonlinearity='relu')
out_good = model_good(x)
print(f"Kaiming init — max activation: {out_good.abs().max().item():.2e}, NaN={torch.isnan(out_good).any().item()}")
"""),

    md("""### Source 10: `torch.where` NaN Trap

**WHY:** `torch.where(cond, A, B)` computes **BOTH** branches before selecting! So `torch.where(x > 0, 1/x, 0)` still evaluates `1/0 = \\infty` even when $x=0$.

**Detect:** Forward looks fine; NaN appears in backward. `detect_anomaly` points to `DivBackward0`.

**Fix:** Clamp denominator before division, or use masked operations.
"""),

    code("""# Source 10: torch.where computes BOTH branches
x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
result = torch.where(x > 0, 1.0 / x, torch.zeros_like(x))
print(f"Forward result: {result}")  # looks OK: [0, 1, 0.5]
loss = result.sum()
loss.backward()
print(f"Gradient at x=0: {x.grad[0].item()}  (inf={torch.isinf(x.grad[0]).item()})")

# Fix: clamp before division
x2 = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
safe_x = x2.clamp(min=1e-8)
result_safe = torch.where(x2 > 0, 1.0 / safe_x, torch.zeros_like(x2))
result_safe.sum().backward()
print(f"Safe gradient at x=0: {x2.grad[0].item()}")
"""),

    md("""### Source 11: `torch.norm` with Zero Vector (Backward)

**WHY:** $\\|x\\|_2 = \\sqrt{\\sum x_i^2}$. Backward: $\\frac{\\partial \\|x\\|}{\\partial x} = \\frac{x}{\\|x\\|}$. When $x = 0$: $\\frac{0}{0} = \\text{NaN}$.

**Detect:** NaN gradient after `.norm()` or `.normalize()` on zero vector.

**Fix:** Add epsilon inside norm: `torch.sqrt((x**2).sum() + 1e-8)`.
"""),

    code("""# Source 11: norm backward on zero vector
x = torch.zeros(5, requires_grad=True)
norm = torch.linalg.norm(x)
print(f"Forward norm: {norm.item()}")
norm.backward()
print(f"Gradient: {x.grad}  (NaN={torch.isnan(x.grad).any().item()})")

x2 = torch.zeros(5, requires_grad=True)
norm_safe = torch.sqrt((x2 ** 2).sum() + 1e-8)
norm_safe.backward()
print(f"Safe gradient: {x2.grad}  (NaN={torch.isnan(x2.grad).any().item()})")
"""),

    md("""### Source 12: Attention Score NaN with Very Long Sequences (FP16)

**WHY:** For seq_len > 4096 with FP16: $QK^\\top / \\sqrt{d}$ can exceed 65504 → Inf before softmax → NaN. Softmax of $[\\infty, \\infty, \\ldots]$ = NaN.

**Detect:** NaN only with long context + FP16; check `scores.max()`.

**Fix:** Flash Attention (fused stable kernel), BF16, or sequence parallelism.
"""),

    code("""# Source 12: attention overflow in FP16 (simulated)
seq_len, d = 8192, 64
Q = torch.randn(1, seq_len, d, dtype=torch.float16) * 2
K = torch.randn(1, seq_len, d, dtype=torch.float16) * 2
scores = torch.matmul(Q, K.transpose(-1, -2)) / (d ** 0.5)
print(f"Max attention score (FP16): {scores.max().item():.1f}")
print(f"Has Inf: {torch.isinf(scores).any().item()}, count>{65504}: {(scores.abs() > 65504).sum().item()}")

# Fix: BF16
scores_bf16 = torch.matmul(Q.to(torch.bfloat16), K.to(torch.bfloat16).transpose(-1, -2)) / (d ** 0.5)
print(f"Max score (BF16): {scores_bf16.max().item():.1f}, Inf={torch.isinf(scores_bf16).any().item()}")
"""),

    md("""### Source 13: Gradient Explosion through Residual Chains

**WHY:** Residual: $h_L = h_0 + \\sum_{l=1}^{L} F_l(h_{l-1})$. Gradient: $\\frac{\\partial \\mathcal{L}}{\\partial h_0} = \\frac{\\partial \\mathcal{L}}{\\partial h_L} \\left(I + \\sum_l \\frac{\\partial F_l}{\\partial h_{l-1}}\\right)$. Skip connections help but Jacobians $> 1$ still compound exponentially with depth + high LR.

**Detect:** Grad norm spikes before NaN; monitor per-layer grad norms.

**Fix:** Pre-LN, grad clip (max_norm=1.0), lower LR, weight decay.
"""),

    code("""# Source 13: gradient growth through deep residual stack
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        nn.init.normal_(self.lin.weight, std=0.5)  # Jacobian can be > 1
    def forward(self, x):
        return x + self.lin(x)

deep = nn.Sequential(*[ResBlock(64) for _ in range(20)])
x = torch.randn(4, 64, requires_grad=True)
out = deep(x)
out.sum().backward()

grad_norms = []
for i, blk in enumerate(deep):
    if blk.lin.weight.grad is not None:
        grad_norms.append(blk.lin.weight.grad.norm().item())

plt.figure(figsize=(10, 4))
plt.plot(grad_norms, 'o-')
plt.xlabel('Layer (0=shallow, 19=deep)'); plt.ylabel('Weight grad norm')
plt.title('Gradient norms through 20-layer residual stack')
plt.tight_layout(); plt.show()
print(f"Shallow grad norm: {grad_norms[0]:.4f}, Deep grad norm: {grad_norms[-1]:.4f}")
print("Fix: Pre-LN, clip_grad_norm_(model.parameters(), 1.0), lower LR")
"""),

    md("""### Source 14: NaN in Mixed Precision Backward (Underflow)

**WHY:** FP16 smallest subnormal $\\approx 5.96 \\times 10^{-8}$. Gradients smaller than this **round to zero**. Downstream layer divides by this zero gradient → NaN in second-order effects or custom ops.

**Detect:** Loss scale stuck at 1; tiny gradients in FP16 master weights not updating FP16 cast weights.

**Fix:** BF16 (wider subnormal range), FP32 master weights (AMP default), gradient accumulation in FP32.
"""),

    code("""# Source 14: FP16 gradient underflow
grad_fp16 = torch.tensor(1e-9, dtype=torch.float16)
grad_fp32 = torch.tensor(1e-9, dtype=torch.float32)
print(f"1e-9 in FP16: {grad_fp16.item()}  (underflowed to 0)")
print(f"1e-9 in FP32: {grad_fp32.item():.2e}")

# Simulate: division by underflowed grad downstream
tiny = torch.tensor(0.0, dtype=torch.float16)  # underflowed grad stored as 0
# Custom op: scale by inverse grad magnitude
if tiny.item() == 0:
    print("Downstream op sees zero grad → potential div-by-zero in custom normalization")
print("Fix: BF16, accumulate grads in FP32, use GradScaler")
"""),

    md("""### Source 15: GAN Training NaN (Discriminator Too Strong)

**WHY:** Generator loss: $-\\log D(G(z))$. When $D(G(z)) \\to 0$ (discriminator perfect): $-\\log(0) = \\infty$ → NaN. Also: $0 \\cdot \\log 0$ in some GAN variants.

**Detect:** NaN after discriminator reaches high accuracy; generator loss spikes.

**Fix:** WGAN-GP (Wasserstein + gradient penalty), spectral norm on D, label smoothing, feature matching, two-timescale update (TTUR).
"""),

    code("""# Source 15: GAN generator loss NaN when D is too strong
def gan_g_loss(D_fake):
    return -torch.log(D_fake)

D_outputs = torch.tensor([0.5, 0.1, 0.01, 1e-8, 0.0])
for d_val in D_outputs:
    loss = gan_g_loss(d_val)
    print(f"D(G(z))={d_val:.1e} → G_loss={loss.item():.4f}  NaN={torch.isnan(loss).item()}")

print("\\nFixes:")
print("  - WGAN-GP: use Wasserstein loss (no log)")
print("  - Label smoothing: D targets in [0.9, 1.0] not 1.0")
print("  - Spectral norm on D: bounds Lipschitz constant")
print("  - Clamp D output: D_fake.clamp(1e-7, 1-1e-7)")
"""),
]

# ---------------------------------------------------------------------------
# Part 4–5: Insert before References in notebook 05
# ---------------------------------------------------------------------------
DEBUG_AND_RECIPES = [
    md("""## 28. Systematic NaN Debugging Methodology

When loss goes NaN, follow this **8-step procedure** (don't guess randomly):

```
1. REPRODUCE  — Find minimum steps to NaN (reduce data, smaller model, single GPU)
2. LOCATE     — torch.autograd.detect_anomaly() → which backward op
3. TRACE      — Register forward hooks on every layer → first NaN layer
4. ISOLATE    — Forward NaN or backward NaN?
                  Forward  → bad data or numerical overflow in computation
                  Backward → gradient explosion or instability in loss function
5. IDENTIFY   — Match against the 15 NaN sources in Section 0d above
6. FIX        — Apply the specific fix for that source
7. VERIFY     — Run for 2× the original NaN step count without NaN
8. HARDEN     — Add permanent NaN hooks / sentinel for production
```

### Decision Tree

```
NaN detected
├── Step 0?        → Check initialization (Source 9)
├── Step 1-100?    → Check LR, warmup, labels (Sources 5, 6)
├── During loss?   → Check log/softmax/KL (Sources 1, 2, 8, 15)
├── In norm layer? → Check LN/BN eps, zero variance (Source 4)
├── In attention?  → Check FP16 overflow, long seq (Source 12)
├── In backward?   → detect_anomaly + NaNTracer (Sources 10, 11)
└── Multi-GPU?     → Check AllReduce propagation (Section 10+)
```
"""),

    md("""## 29. NaNTracer — Complete Layer-by-Layer Utility

Attach `NaNTracer` to any model to find the **exact layer and direction** (forward vs backward) of the first NaN/Inf.
"""),

    code("""class NaNTracer:
    \"\"\"Attach to model; traces exact layer and direction of first NaN.\"\"\"
    def __init__(self, model):
        self.model = model
        self.first_nan = None
        self._hooks = []
        for name, mod in model.named_modules():
            if name == '':
                continue
            self._hooks.append(mod.register_forward_hook(self._make_fwd(name)))
            self._hooks.append(mod.register_full_backward_hook(self._make_bwd(name)))

    def _make_fwd(self, name):
        def hook(mod, inp, out):
            if self.first_nan:
                return
            outs = [out] if isinstance(out, torch.Tensor) else (
                list(out) if isinstance(out, tuple) else [])
            for o in outs:
                if isinstance(o, torch.Tensor) and (torch.isnan(o).any() or torch.isinf(o).any()):
                    n_nan = torch.isnan(o).sum().item()
                    n_inf = torch.isinf(o).sum().item()
                    self.first_nan = f"FORWARD in '{name}': {n_nan} NaN, {n_inf} Inf"
                    print(f"*** FIRST NaN: {self.first_nan}")
                    for i, inp_t in enumerate(inp):
                        if isinstance(inp_t, torch.Tensor):
                            print(f"    Input[{i}]: shape={inp_t.shape}, "
                                  f"has_nan={torch.isnan(inp_t).any().item()}, "
                                  f"range=[{inp_t.min().item():.4f}, {inp_t.max().item():.4f}]")
        return hook

    def _make_bwd(self, name):
        def hook(mod, grad_in, grad_out):
            if self.first_nan:
                return
            for i, g in enumerate(grad_out):
                if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                    self.first_nan = f"BACKWARD in '{name}': grad_out[{i}] has NaN/Inf"
                    print(f"*** FIRST NaN: {self.first_nan}")
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()

    def report(self):
        if self.first_nan:
            print(f"\\nNaN Source: {self.first_nan}")
        else:
            print("\\nNo NaN detected in this forward+backward pass")
        return self.first_nan


# Demo: trace NaN through a small network
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()
    def forward(self, x):
        h = F.relu(self.fc1(x))
        # Zero fc2 weights → zero output → div by zero norm
        return self.fc2(h) / h.norm(dim=-1, keepdim=True)

model = BadNet()
tracer = NaNTracer(model)
x = torch.randn(2, 4)
try:
    with torch.autograd.detect_anomaly():
        out = model(x)
        out.sum().backward()
except RuntimeError:
    pass
tracer.report()
tracer.remove()
"""),

    md("""## 30. NaN Prevention Recipes — Model-Type Summary

| Model Type | Common NaN Source | Prevention Recipe |
|------------|-------------------|-------------------|
| CLIP / Contrastive | τ collapse, FP16 logit overflow | Clamp τ≥0.01, BF16, grad clip 1.0 |
| Transformer LM | Attention overflow, Post-LN instability | Pre-LN, flash attention, BF16 |
| Image Captioning | log(0) in CE, empty caption | Label smoothing, min caption len=1 |
| VQA | Soft CE with zero probs | Clamp probs, use log_softmax |
| LoRA Finetuning | Large α/r scaling | α=r, grad clip 1.0 |
| Diffusion | Noise schedule edge cases | Clamp σ, stable loss |
| GAN | −log(D(G(z))) with D→0 | WGAN-GP, spectral norm |
| Multi-GPU DDP | AllReduce NaN propagation | GradientSentinel, data validation |
| FSDP | Mixed precision casting | bf16/fp32/fp32 combination |
| DeepSpeed ZeRO | Sharded grad NaN localization | safe_get_full_grad, grad clipping |

**Golden rules:**
1. Always use `detect_anomaly()` first when debugging
2. Prefer BF16 over FP16 on Ampere+ GPUs
3. Always `clip_grad_norm_(..., 1.0)` for transformers
4. Validate data before forward (no NaN inputs, valid labels)
5. Monitor loss scale and grad norms proactively
"""),
]

# ---------------------------------------------------------------------------
# Notebook 06: distributed NaN prevention before References
# ---------------------------------------------------------------------------
NB06_DISTRIBUTED_RECIPES = [
    md("""## 21. Distributed NaN Prevention Recipes

| Scenario | NaN Mechanism | Distributed Fix |
|----------|---------------|-----------------|
| DDP AllReduce | One rank NaN → all ranks NaN | GradientSentinel before sync; validate per-rank batch |
| FSDP sharding | NaN hidden in param shard | MixedPrecision(bf16, fp32, fp32); inspect full grad |
| ZeRO Stage 3 | Partial NaN in grad partition | `safe_get_full_grad`; grad clipping in DeepSpeed config |
| LR not scaled | Effective batch ↑, LR same → spike | $\\eta_{new} = \\eta_{base} \\times B_{global}/B_{base}$ |
| Short warmup | Large global batch, tiny warmup → NaN step 1–10 | warmup_steps $\\propto G$ |
| Grad accum + AMP | Unscale inside micro-loop | Unscale **once** after $K$ accumulation steps |
| BatchNorm DDP | $B_{local} < 4$ → zero variance | SyncBatchNorm or GroupNorm/LayerNorm |
| Pipeline parallel | NaN in stage 3, hard to locate | Log micro-batch ID + stage rank; per-stage NaN hooks |
| Tensor parallel | Vocab-parallel CE overflow | Stable log-sum-exp cross-entropy (Megatron-style) |
| NCCL hang | Rank desync (not NaN but related) | Same collectives all ranks; NCCL_DEBUG=INFO |

### Launch Checklist (copy before every multi-GPU run)

```bash
# 1. Scale LR
# lr = base_lr * num_gpus * local_batch / base_batch

# 2. Scale warmup
# warmup_steps = max(500, 100 * num_gpus)

# 3. Debug env (remove in production)
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 4. Precision
# autocast(dtype=torch.bfloat16)  # NOT float16 on A100+

# 5. Safety
# clip_grad_norm_(model.parameters(), 1.0)
# GradientSentinel.sanitize(model) before optimizer.step()
```
"""),

    md("""## 22. Comprehensive NaN Prevention Table (All Model Types × Distributed)

| Model Type | Single-GPU NaN Source | Distributed Amplifier | Full Prevention Recipe |
|------------|----------------------|----------------------|------------------------|
| CLIP / Contrastive | τ collapse, log(0) | FP16 overflow × AllReduce | Clamp τ≥0.01, BF16, grad clip 1.0, LR scale × G |
| Transformer LM | Attention FP16 overflow | Long seq × tensor parallel | Pre-LN, flash attention, BF16, vocab-parallel CE |
| Image Captioning | log(0) in CE | Empty caption on one rank | Label smoothing, min len=1, DataValidator per rank |
| VQA | Soft CE zero probs | — | log_softmax, clamp probs, sentinel |
| LoRA Finetuning | Large α/r | FSDP dtype mismatch | α=r, MixedPrecision(bf16/fp32/fp32), clip 1.0 |
| Diffusion | σ schedule edge | Pipeline stage NaN | Clamp σ∈[1e-4, 1], per-stage hooks |
| GAN | −log(D(G(z))) | D stronger on one GPU | WGAN-GP, spectral norm, sync D/G updates |
| Multi-GPU DDP | Any local NaN | AllReduce poisoning | Sentinel + validate before backward |
| FSDP | Mixed precision | Wrong reduce dtype | bf16/fp32/fp32 only |
| DeepSpeed ZeRO | Sharded NaN | Hidden in partition | safe_get_full_grad, gradient_clipping=1.0 |

See notebook **05** Section 0 for IEEE 754 fundamentals, 15 NaN sources, `detect_anomaly()`, and `NaNTracer`.
"""),
]


def insert_at(cells, index, new_cells):
    return cells[:index] + new_cells + cells[index:]


def insert_before_references(nb_path: Path, new_cells: list):
    with open(nb_path) as f:
        nb = json.load(f)
    cells = nb["cells"]
    ref_idx = next(i for i, c in enumerate(cells) if "References" in "".join(c.get("source", [])))
    nb["cells"] = cells[:ref_idx] + new_cells + cells[ref_idx:]
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
        f.write("\n")
    print(f"Inserted {len(new_cells)} cells before References in {nb_path.name} → {len(nb['cells'])} total")


def insert_after_imports(nb_path: Path, new_cells: list, import_idx: int = 2):
    """Insert after imports cell (default index 2)."""
    with open(nb_path) as f:
        nb = json.load(f)
    cells = nb["cells"]
    nb["cells"] = insert_at(cells, import_idx + 1, new_cells)
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
        f.write("\n")
    print(f"Inserted {len(new_cells)} cells after imports in {nb_path.name} → {len(nb['cells'])} total")


if __name__ == "__main__":
    insert_after_imports(NB05, IEEE_AND_SOURCES, import_idx=2)
    insert_before_references(NB05, DEBUG_AND_RECIPES)
    insert_before_references(NB06, NB06_DISTRIBUTED_RECIPES)
    print("Done.")
