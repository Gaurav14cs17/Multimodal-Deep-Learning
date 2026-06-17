#!/usr/bin/env python3
"""Expand Module 07 notebooks 05 and 06 with multi-GPU NaN debugging content."""
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


def insert_before_references(nb_path: Path, new_cells: list, new_refs: str):
    with open(nb_path) as f:
        nb = json.load(f)
    cells = nb["cells"]
    ref_idx = next(i for i, c in enumerate(cells) if "References" in "".join(c.get("source", [])))
    nb["cells"] = cells[:ref_idx] + new_cells + [md(new_refs)]
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
        f.write("\n")
    print(f"Updated {nb_path.name}: {len(nb['cells'])} cells")


# ---------------------------------------------------------------------------
# Notebook 05 — new cells (before References)
# ---------------------------------------------------------------------------
NB05_NEW = [
    md("""## 10. Multi-GPU NaN — Why Single-GPU Fixes Don't Always Work

On one GPU, NaN usually means bad numerics locally. On **multi-GPU**, a NaN on **one rank** poisons **all ranks** via AllReduce:

$$
g = \\frac{1}{G}\\sum_{r=1}^{G} g_r \\quad\\Rightarrow\\quad \\text{if any } g_r = \\text{NaN},\\; g = \\text{NaN}
$$

This section covers the **10 most common multi-GPU NaN root causes** with runnable PyTorch demos (CPU-safe; patterns apply on CUDA + DDP).
"""),

    md("""## 11. FP16/BF16 Overflow in AMP (Most Common Multi-GPU NaN)

**FP16 max representable** $\\approx 65504$. Values above overflow to $\\infty$; $\\infty - \\infty \\to$ NaN.

With AMP, loss is scaled before backward:

$$
\\text{scaled\\_loss} = s \\cdot \\mathcal{L}, \\quad s \\text{ starts high (e.g. } 65536\\text{)}
$$

If $\\mathcal{L} \\cdot s > 65504$ in FP16 forward activations → overflow → NaN.

**BF16** shares FP32 exponent range (no 65504 cliff) but lower mantissa precision — preferred on A100/H100 for multi-GPU training.
"""),

    code("""# FP16 overflow demo vs BF16 stability
import torch

FP16_MAX = torch.finfo(torch.float16).max
print(f"FP16 max: {FP16_MAX.item():.0f}")

# Simulate CLIP-like logits / tau in FP16
tau = torch.tensor(0.001, dtype=torch.float16)
similarities = torch.tensor([30.0, 25.0, 20.0], dtype=torch.float16)
logits_fp16 = similarities / tau
print(f"logits/tau in FP16: {logits_fp16}  (max={logits_fp16.max().item():.0f})")
print(f"Any Inf? {torch.isinf(logits_fp16).any().item()}")

# Same in BF16 — same exponent range as FP32
logits_bf16 = (similarities.float() / tau.float()).to(torch.bfloat16)
print(f"logits/tau in BF16 max: {logits_bf16.max().item():.0f}, Inf? {torch.isinf(logits_bf16).any().item()}")
"""),

    md("""## 12. Dynamic Loss Scaling — Math & Failure Mode

PyTorch `GradScaler` adjusts $s$ each step:

- **NaN/Inf in grads** → skip step, $s \\leftarrow s / 2$
- **$N$ consecutive good steps** → $s \\leftarrow \\min(2s, s_{max})$

When $s \\to 1$ after many skips, effective update $\\approx \\eta \\cdot g / s$ becomes tiny → **training stops learning** (looks alive, loss flatlines, then eventual NaN from stale numerics).
"""),

    code("""# Simulate dynamic loss scale behavior (PyTorch GradScaler logic)
class LossScaleMonitor:
    \"\"\"Track AMP loss scale over time, alert when dropping.\"\"\"
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, alert_threshold=1.0):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.alert_threshold = alert_threshold
        self.history = []
        self.good_steps = 0

    def step(self, had_nan: bool):
        if had_nan:
            self.scale *= self.backoff_factor
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self.good_steps = 0
        self.history.append(self.scale)
        if self.scale <= self.alert_threshold:
            print(f"ALERT: loss scale dropped to {self.scale:.1f} — check LR and data quality")
        return self.scale

monitor = LossScaleMonitor()
# Simulate 5 NaN events then recovery
for step in range(50):
    had_nan = step in {3, 4, 5, 6, 7}
    s = monitor.step(had_nan)
    if step < 12 or step > 45:
        print(f"step {step:2d}: scale={s:8.1f}  nan={had_nan}")

plt.figure(figsize=(10, 4))
plt.plot(monitor.history)
plt.axhline(65504, color='r', ls='--', label='FP16 max (65504)')
plt.xlabel('Step'); plt.ylabel('Loss scale'); plt.title('Dynamic Loss Scale After NaN Skips')
plt.legend(); plt.tight_layout(); plt.show()
"""),

    md("""## 13. Gradient Accumulation + AMP — Wrong vs Right

When accumulating $K$ micro-batches, **unscale once** after all micro-steps:

**Wrong:** `scaler.unscale_(optimizer)` inside the micro-step loop → divides grads by $s$ multiple times → wrong magnitude → NaN.

**Right:** accumulate scaled grads for $K$ steps, then `scaler.unscale_` once, clip, `scaler.step`.
"""),

    code("""# Wrong vs right gradient accumulation + AMP pattern (conceptual, CPU)
import torch
from torch.cuda.amp import GradScaler

def wrong_pattern(model, optimizer, micro_batches, accum_steps=4):
    \"\"\"ANTI-PATTERN: unscale inside micro-step loop.\"\"\"
    scaler = GradScaler()
    optimizer.zero_grad()
    for i, batch in enumerate(micro_batches):
        with torch.cuda.amp.autocast(enabled=False):  # CPU demo
            loss = model(batch).sum() / accum_steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # WRONG: called every micro-step
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

def right_pattern(model, optimizer, micro_batches, accum_steps=4):
    \"\"\"CORRECT: unscale once after accumulation.\"\"\"
    scaler = GradScaler()
    optimizer.zero_grad()
    for i, batch in enumerate(micro_batches):
        with torch.cuda.amp.autocast(enabled=False):
            loss = model(batch).sum() / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)  # RIGHT: once per optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
batches = [torch.randn(2, 4) for _ in range(8)]
right_pattern(model, opt, batches)
print("Correct pattern: unscale AFTER all micro-steps in accumulation window")
"""),

    md("""## 14. Cross-GPU Gradient Sync — NaN Propagation

If rank 2 produces NaN gradients, AllReduce average is NaN on **every** rank.

**Fix:** gradient sentinel — detect NaN **before** sync, zero out bad grads on that rank (or skip batch).
"""),

    code("""class GradientSentinel:
    \"\"\"Check gradients before AllReduce, replace NaN/Inf with zero.\"\"\"
    def __init__(self, rank=0):
        self.rank = rank
        self.nan_events = 0

    def sanitize(self, model) -> bool:
        had_bad = False
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                had_bad = True
                self.nan_events += 1
                print(f"[rank {self.rank}] NaN/Inf in {name} — zeroing grad")
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        return had_bad

def simulate_allreduce(grads_per_rank):
    \"\"\"Simulate mean AllReduce across virtual GPUs.\"\"\"
    stacked = torch.stack(grads_per_rank)
    if torch.isnan(stacked).any():
        print("AllReduce result: NaN (poisoned by at least one rank)")
    return stacked.mean(dim=0)

# Virtual 4-GPU: rank 2 has NaN gradient
grads = [torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.5]),
         torch.tensor([float('nan'), 3.0]), torch.tensor([1.2, 0.8])]
print("Before sentinel:", simulate_allreduce(grads))

# Apply sentinel on rank 2
grads[2] = torch.nan_to_num(grads[2], nan=0.0)
print("After sentinel: ", simulate_allreduce(grads))
"""),

    md("""## 15. Loss Spike → NaN Pattern on Multi-GPU

Typical timeline when one GPU sees a bad batch:

```
Step 1000: loss = 2.3   (normal)
Step 1001: loss = 15.7  (spike — GPU 2 adversarial batch)
Step 1002: loss = 847.2 (exploding — large grad update)
Step 1003: loss = NaN   (dead)
```

After AllReduce, the spike gradient is averaged:

$$
g = \\frac{1}{G}\\left(g_1 + \\cdots + g_{\\text{bad}} + \\cdots + g_G\\right)
$$

Even one large $g_{\\text{bad}}$ can dominate if not clipped. **Fix:** per-rank loss clipping, skip batches with loss $> \\tau$, monitor grad norm per rank.
"""),

    code("""# Simulate loss spike propagation through gradient averaging
G = 4
normal_grads = [torch.randn(100) * 0.01 for _ in range(G)]
bad_grad = torch.randn(100) * 50.0  # spike on rank 2
normal_grads[2] = bad_grad

avg_before_clip = torch.stack(normal_grads).mean(0)
print(f"Global grad norm before clip: {avg_before_clip.norm():.2f}")

# Per-rank clipping before AllReduce
max_norm = 1.0
clipped = []
for g in normal_grads:
    norm = g.norm()
    if norm > max_norm:
        g = g * (max_norm / norm)
    clipped.append(g)
avg_after_clip = torch.stack(clipped).mean(0)
print(f"Global grad norm after per-rank clip: {avg_after_clip.norm():.2f}")

# Per-GPU loss clipping
losses = torch.tensor([2.3, 2.1, 15.7, 2.4])  # rank 2 spike
loss_threshold = 10.0
for r, loss in enumerate(losses):
    if loss > loss_threshold:
        print(f"Rank {r}: skip batch (loss={loss:.1f} > {loss_threshold})")
"""),

    md("""## 16. BatchNorm NaN in Multi-GPU

Per-GPU batch norm uses **local** batch statistics:

$$
\\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}
$$

When per-GPU batch is tiny ($B_{\\text{local}} < 4$), $\\sigma^2 \\approx 0$ on one GPU → division instability → NaN → sync poisons all ranks.

**Fixes:** `SyncBatchNorm`, or replace with GroupNorm/LayerNorm when $B_{\\text{local}}$ is small.
"""),

    code("""# BatchNorm NaN with tiny per-GPU batch
def demo_bn_nan(batch_per_gpu=2):
    bn = nn.BatchNorm1d(8)
    bn.train()
    x = torch.ones(batch_per_gpu, 8) * 5.0  # zero variance within batch
    x[:, 0] += torch.randn(batch_per_gpu) * 1e-8
    out = bn(x)
    return torch.isnan(out).any().item(), out

nan_tiny, _ = demo_bn_nan(2)
print(f"BatchNorm with batch=2: NaN={nan_tiny}")

# SyncBatchNorm aggregates stats across GPUs (simulated with larger effective batch)
sync_bn = nn.SyncBatchNorm(8)
sync_bn.train()
# Simulate merged batch from 4 GPUs
merged = torch.randn(8, 8)  # effective batch = 8
out_sync = sync_bn(merged)
print(f"SyncBatchNorm with effective batch=8: NaN={torch.isnan(out_sync).any().item()}")

# GroupNorm fix — no batch-dim dependency
gn = nn.GroupNorm(4, 8)
out_gn = gn(torch.ones(2, 8) * 5.0)
print(f"GroupNorm with batch=2: NaN={torch.isnan(out_gn).any().item()}")
"""),

    md("""## 17. NCCL Timeout & Hangs (Multi-GPU Failure Mode)

Not always NaN, but the **#2 multi-GPU failure** after numerics:

- Ranks finish at different speeds → NCCL timeout
- **Conditional branching** per rank (`if rank == 0: ...`) without barriers → deadlock
- Debug: `NCCL_DEBUG=INFO`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`

Environment variables:

```
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
```

**Rule:** every rank must execute the same collective ops in the same order.
"""),

    md("""## 18. Gradient Norm Explosion on Specific Ranks

Data imbalance → one rank sees harder examples → higher local grad norm:

$$
\\lVert g_r \\rVert \\gg \\lVert g_{r'} \\rVert \\quad \\text{for some } r
$$

**Fix:** compute per-rank norm, clip **before** AllReduce, log variance across ranks.
"""),

    code("""# Per-rank gradient norm variance simulation
G = 8
torch.manual_seed(0)
grad_norms = []
for rank in range(G):
    # Rank 5 gets harder data (larger gradients)
    scale = 5.0 if rank == 5 else 1.0
    g = torch.randn(1000) * 0.01 * scale
    grad_norms.append(g.norm().item())

print("Per-rank grad norms:", [f"{n:.4f}" for n in grad_norms])
print(f"Mean={np.mean(grad_norms):.4f}, Std={np.std(grad_norms):.4f}, Max/Min={max(grad_norms)/min(grad_norms):.1f}x")

# Clip per rank before sync
max_norm = 1.0
clipped_norms = [min(n, max_norm) for n in grad_norms]
print("After per-rank clip:", [f"{n:.4f}" for n in clipped_norms])
"""),

    md("""## 19. Mixed Precision Master Weight Desync

AMP keeps **FP32 master weights** $W_{32}$ and casts to FP16 $W_{16}$ for forward.

Smallest FP16 increment: $\\Delta_{16} \\approx 2^{-24} \\approx 5.96 \\times 10^{-8}$.

When $\\lvert \\eta \\cdot g \\rvert < \\Delta_{16}$:

$$
W_{16} = \\text{cast}(W_{32}) \\quad \\text{does not change after update}
$$

Stagnation in FP16 weights → activations drift → eventual overflow/NaN. **Always use FP32 master weights** (PyTorch AMP default with `GradScaler`).
"""),

    code("""# FP16 weight stagnation demo
W32 = torch.tensor([1.0], dtype=torch.float32)
lr, grad = 1e-9, 1.0  # tiny update
W32_new = W32 - lr * grad
W16_before = W32.half()
W16_after = W32_new.half()
print(f"FP32 update: {W32.item():.10f} -> {W32_new.item():.10f}")
print(f"FP16 before: {W16_before.item():.6f}, after: {W16_after.item():.6f}, changed: {W16_before.item() != W16_after.item()}")

# Many tiny updates accumulate in FP32 but FP16 appears frozen
W32 = torch.tensor([1.0], dtype=torch.float32)
for _ in range(10000):
    W32 = W32 - 1e-7  # visible in FP32
print(f"After 10k tiny updates: FP32={W32.item():.6f}, FP16={W32.half().item():.6f}")
"""),

    md("""## 20. Reproducibility Across GPUs (Debugging Confusion)

Non-deterministic ops make multi-GPU bugs **hard to reproduce**:

- `atomicAdd` in CUDA backward
- `cudnn.benchmark = True`
- Different seeds per rank → different data order

```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Use **same base seed + rank offset** for DistributedSampler: `seed + rank`.
"""),

    code("""# Reproducibility settings (apply before training)
import os

def set_deterministic(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Deterministic mode note: {e}")

set_deterministic(42)
a = torch.randn(3)
set_deterministic(42)
b = torch.randn(3)
print(f"Same seed -> same tensor: {torch.equal(a, b)}")
"""),

    md("""## 21. Complete Multi-GPU NaN Debugging Checklist

```
NaN on multi-GPU?
├── Which step? (early vs late)
│   ├── Step 0-10: Initialization problem
│   │   └── Check: init scale, LR warmup
│   ├── Step 10-1000: LR/optimizer issue
│   │   └── Check: LR schedule, loss scale
│   └── Step 1000+: Data/numerical issue
│       └── Check: batch quality, softmax overflow
├── Which rank first?
│   ├── All ranks simultaneously: global issue (LR, loss function)
│   └── Single rank first: data issue on that rank, BatchNorm
├── Using AMP?
│   ├── FP16: likely overflow → try BF16
│   └── BF16: rare overflow → check loss function
├── Loss scale dropping to 1?
│   └── Too many NaN skips → lower LR, check data
└── Grad norm before NaN?
    └── If spike: clip grads, check data quality
```
"""),

    md("""## 22. NaN Prevention Toolkit

Complete utilities for production multi-GPU training: layer hooks, gradient sentinel, loss scale monitor, rank-aware logging, and a safe training loop skeleton.
"""),

    code("""class NaNDetector:
    \"\"\"Hook that monitors every layer for NaN/Inf in forward/backward.\"\"\"
    def __init__(self, model, rank=0):
        self.rank = rank
        self.events = []
        self._handles = []
        for name, module in model.named_modules():
            if name == '':
                continue
            self._handles.append(module.register_forward_hook(self._fwd_hook(name)))
            self._handles.append(module.register_full_backward_hook(self._bwd_hook(name)))

    def _fwd_hook(self, name):
        def hook(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else None
            if t is not None and (torch.isnan(t).any() or torch.isinf(t).any()):
                self.events.append(('forward', name))
                print(f"[rank {self.rank}] FWD NaN/Inf in {name}")
        return hook

    def _bwd_hook(self, name):
        def hook(mod, grad_in, grad_out):
            for g in grad_out:
                if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                    self.events.append(('backward', name))
                    print(f"[rank {self.rank}] BWD NaN/Inf in {name}")
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


class RankAwareLogger:
    \"\"\"Per-GPU logging for distributed debugging.\"\"\"
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

    def log(self, msg, main_only=False):
        if main_only and self.rank != 0:
            return
        print(f"[rank {self.rank}/{self.world_size}] {msg}")

    def log_all_ranks(self, values: dict):
        if self.rank == 0:
            print(" | ".join(f"r{k}={v:.4f}" for k, v in sorted(values.items())))


class SafeTrainingLoop:
    \"\"\"Training loop skeleton with NaN protections (single-process demo).\"\"\"
    def __init__(self, model, optimizer, rank=0, world_size=1, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.max_grad_norm = max_grad_norm
        self.sentinel = GradientSentinel(rank)
        self.scale_monitor = LossScaleMonitor()
        self.logger = RankAwareLogger(rank, world_size)
        self.detector = NaNDetector(model, rank)

    def train_step(self, loss: torch.Tensor) -> bool:
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.log(f"Bad loss={loss.item()}, skipping step")
            self.scale_monitor.step(had_nan=True)
            return False
        self.optimizer.zero_grad()
        loss.backward()
        self.sentinel.sanitize(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scale_monitor.step(had_nan=False)
        return True

# Quick demo
m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
loop = SafeTrainingLoop(m, opt)
x = torch.randn(4, 10)
ok = loop.train_step(m(x).sum())
print(f"Step OK: {ok}, loss scale: {loop.scale_monitor.scale}")
loop.detector.remove()
"""),

    md("""## 23. Scenario 1 — CLIP NaN at Step 500 on 4 GPUs

**Symptoms:** NaN around step 500, 4× GPU DDP, FP16, learnable temperature, no grad clip.

**Diagnosis:** $\\tau \\to 0.001$ → logits $s/\\tau > 65504$ in FP16.

**Fix:** clamp $\\tau \\geq 0.01$, switch to BF16, add `clip_grad_norm_(..., 1.0)`.
"""),

    code("""# Scenario 1: CLIP temperature collapse in FP16
class LearnableTemperatureCLIP(nn.Module):
    def __init__(self, dim=4, tau_init=0.07, tau_min=0.01):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init)))
        self.tau_min = tau_min

    def forward(self, sim):
        tau = self.log_tau.exp().clamp(min=self.tau_min)  # FIX: clamp
        logits = sim / tau
        labels = torch.arange(sim.size(0))
        return F.cross_entropy(logits, labels), tau

def train_clip_scenario(use_clamp=True, dtype=torch.float16):
    torch.manual_seed(42)
    model = LearnableTemperatureCLIP(tau_min=0.01 if use_clamp else 1e-6)
    opt = torch.optim.Adam([model.log_tau], lr=0.05)
    S = torch.randn(4, 4) * 0.5
    nan_step = None
    for step in range(200):
        opt.zero_grad()
        sim = S.to(dtype)
        loss, tau = model(sim)
        if torch.isnan(loss):
            nan_step = step
            break
        loss.backward()
        opt.step()
    return nan_step, model.log_tau.exp().item()

nan_no_clamp, tau_bad = train_clip_scenario(use_clamp=False)
nan_clamp, tau_good = train_clip_scenario(use_clamp=True)
print(f"Without clamp: NaN at step {nan_no_clamp}, final tau={tau_bad:.6f}")
print(f"With clamp:    NaN at step {nan_clamp}, final tau={tau_good:.6f}")
"""),

    md("""## 24. Scenario 2 — Fine for 10k Steps, Then Sudden NaN on 8 GPUs

**Symptoms:** AMP FP16, loss scale stuck at 1, grad norms creeping up.

**Root cause:** weights grow → activations grow → FP16 overflow.

**Fix:** weight decay, grad clip, switch to BF16, monitor loss scale.
"""),

    code("""# Scenario 2: monitor loss scale + grad norms over training
class GrowingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return (self.w * x).pow(2).mean()

model = GrowingModel()
opt = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=0.0)  # no WD -> grows
monitor = LossScaleMonitor(init_scale=65536.0)
grad_norms, scales = [], []

for step in range(300):
    x = torch.randn(32)
    opt.zero_grad()
    loss = model(x)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([model.w], float('inf')).item()
    grad_norms.append(gn)
    opt.step()
    had_nan = np.isnan(gn) or gn > 1e4
    scales.append(monitor.step(had_nan))
    if had_nan:
        print(f"Step {step}: grad_norm={gn:.2e}, w={model.w.item():.4f}")
        break

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(grad_norms); ax[0].set_title('Grad norm creep (no weight decay)'); ax[0].set_xlabel('Step')
ax[1].plot(scales); ax[1].set_title('Loss scale'); ax[1].set_xlabel('Step')
plt.tight_layout(); plt.show()
print("Fix: add weight_decay=0.01, clip_grad_norm=1.0, use bfloat16")
"""),

    md("""## 25. Scenario 3 — NaN Only on Rank 3 (Then All Ranks)

**Symptoms:** rank 3 logs NaN first; shortly after all ranks NaN.

**Root cause:** corrupted input on rank 3 → NaN embedding → AllReduce poisons everyone.

**Fix:** validate data, check for NaN **before** backward, skip bad batches.
"""),

    code("""class DataValidator:
    \"\"\"Validate batches before forward; skip corrupted samples.\"\"\"
    def __init__(self, rank=0):
        self.rank = rank
        self.skipped = 0

    def validate(self, batch: torch.Tensor) -> bool:
        if torch.isnan(batch).any() or torch.isinf(batch).any():
            self.skipped += 1
            print(f"[rank {self.rank}] SKIP corrupted batch (total skipped={self.skipped})")
            return False
        if batch.abs().max() > 1e4:
            self.skipped += 1
            print(f"[rank {self.rank}] SKIP extreme values max={batch.abs().max():.2e}")
            return False
        return True

def embed(x):
    return x @ torch.tensor([[1.0, float('nan')], [0.5, 0.5]])  # corrupted path

validator = DataValidator(rank=3)
batches = [torch.randn(4, 2) for _ in range(5)]
batches[2][0, 1] = float('nan')  # corrupted batch on rank 3

for i, batch in enumerate(batches):
    if not validator.validate(batch):
        continue
    out = embed(batch)
    if torch.isnan(out).any():
        print(f"Would poison AllReduce at batch {i} without validator")
    else:
        print(f"Batch {i} OK")
"""),

    md("""## 26. Scenario 4 — LLaVA Finetune NaN with LoRA + Multi-GPU

**Symptoms:** BF16 + LoRA rank 16 + FSDP, NaN in vision projector.

**Root cause:** LoRA scaling $\\alpha/r$ too high → large delta weights → overflow in projector.

**Fix:** lower $\\alpha$, BF16 (not FP16), `clip_grad_norm=1.0`, sane FSDP mixed precision.
"""),

    code("""# Scenario 4: LoRA scaling and safe config
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank=16, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_f, out_f, bias=False)
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

def lora_forward_norm(alpha, rank=16):
    layer = LoRALinear(512, 512, rank=rank, alpha=alpha)
    x = torch.randn(8, 512)
    out = layer(x)
    return out.abs().max().item()

for alpha in [16, 64, 256]:
    mx = lora_forward_norm(alpha)
    print(f"alpha={alpha:3d}, alpha/r={alpha/16:.1f}, max activation={mx:.2f}")

print('Safe LLaVA + LoRA + FSDP config:')
print('  - alpha=16, rank=16 (scaling=1.0)')
print('  - dtype=bfloat16')
print('  - clip_grad_norm=1.0')
print('  - FSDP MixedPrecision(param_dtype=bfloat16, reduce_dtype=float32)')
"""),

    md("""## 27. Updated Instability Fix Table (Multi-GPU)

| Issue | Fix | Priority |
|-------|-----|----------|
| FP16 overflow | BF16 + loss scaling | Immediate |
| NaN on one rank | Gradient sentinel + data validation | Immediate |
| Loss scale → 1 | Lower LR, fix data, fewer NaN batches | High |
| BatchNorm NaN | SyncBatchNorm or GroupNorm | High |
| Grad accum + AMP | Unscale once after accumulation | High |
| CLIP NaN | Clamp τ, BF16, grad clip | High |
| NCCL hang | Same collectives all ranks; NCCL_DEBUG | Medium |
"""),
]

NB05_REFS = """## References & Further Reading

### Papers
- Micikevicius et al. (2018) — Mixed Precision Training — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Ott et al. (2019) — fairseq: FP16 Training — [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Xiong et al. (2020) — On Layer Normalization in Transformers — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Yang et al. (2022) — Tensor Programs V (μP) — [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- Miyato et al. (2018) — Spectral Normalization — [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
- Goyal et al. (2017) — Accurate, Large Minibatch SGD — [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

### Blogs & Docs
- [Lilian Weng — Training Large Neural Networks](https://lilianweng.github.io/posts/2021-09-25-train-compute/)
- [HuggingFace — Debugging Mixed Precision](https://huggingface.co/docs/transformers/perf_train_gpu_one#mixed-precision-training)
- [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA — Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
"""

# ---------------------------------------------------------------------------
# Notebook 06 — new cells
# ---------------------------------------------------------------------------
NB06_NEW = [
    md("""## 8. Multi-GPU NaN in Distributed Training

Distributed training **amplifies** numerical failures: larger effective batch without LR scaling → divergence; FSDP/ZeRO sharding → harder NaN localization; gradient checkpointing + AMP + FSDP → subtle wrapper ordering bugs.

This section covers DeepSpeed ZeRO, FSDP mixed precision, LR scaling, warmup scaling, and distributed debugging scenarios.
"""),

    md("""## 9. Learning Rate Linear Scaling Rule

When scaling from base batch $B_{\\text{base}}$ to global batch $B_{\\text{global}}$ across $G$ GPUs:

$$
\\eta_{\\text{new}} = \\eta_{\\text{base}} \\times \\frac{B_{\\text{global}}}{B_{\\text{base}}}
$$

**Example:** 4 GPUs × batch 32 = 128 global batch, base batch 32 → scale LR by **4×**.

Without this: effective batch grows but LR stays same → sharp loss spikes → NaN.

**When it breaks:** batch $> 8k$ → use LARS/LAMB (You et al. 2017/2020).
"""),

    code("""# LR linear scaling calculator
def scaled_lr(base_lr, base_batch, num_gpus, local_batch):
    global_batch = num_gpus * local_batch
    return base_lr * (global_batch / base_batch)

base_lr, base_batch = 1e-4, 32
for gpus in [1, 2, 4, 8]:
    lr = scaled_lr(base_lr, base_batch, gpus, local_batch=32)
    print(f"{gpus} GPUs x 32 batch -> global={gpus*32:3d}, lr={lr:.6f} ({lr/base_lr:.0f}x base)")
"""),

    md("""## 10. Warmup MUST Scale with GPU Count

More GPUs → larger effective batch → optimizer sees larger gradient variance early.

Rule of thumb:

$$
\\text{warmup\\_steps} \\propto G \\quad \\text{or} \\quad \\text{warmup\\_steps} \\propto \\sqrt{G}
$$

Without adequate warmup: **NaN at steps 1–10** on multi-GPU is often misdiagnosed as "bad model."
"""),

    code("""# Warmup schedules for different GPU counts
def warmup_lr(step, warmup_steps, base_lr, scaled_lr_val):
    if step < warmup_steps:
        return scaled_lr_val * (step + 1) / warmup_steps
    return scaled_lr_val

base_lr = 1e-4
local_batch = 32
steps = np.arange(0, 50)

fig, ax = plt.subplots(figsize=(10, 5))
for gpus in [1, 4, 8]:
    global_b = gpus * local_batch
    lr_scaled = base_lr * (global_b / 32)
    warmup = max(500, 100 * gpus)  # scale warmup with GPU count
    curve = [warmup_lr(s, warmup, base_lr, lr_scaled) for s in steps]
    ax.plot(steps, curve, label=f'{gpus} GPUs (warmup={warmup}, lr={lr_scaled:.2e})')

ax.set_xlabel('Step'); ax.set_ylabel('Learning rate')
ax.set_title('Warmup must grow with GPU count'); ax.legend(); plt.tight_layout(); plt.show()
"""),

    md("""## 11. DeepSpeed ZeRO NaN Debugging

ZeRO partitions optimizer states (Stage 1), gradients (Stage 2), parameters (Stage 3). NaN in one partition shard is harder to locate.

**Issues:**
- Gradient partitioning → partial NaN not visible on all ranks
- `GatheredParameters` context errors masking NaN source

**Fix:** use `deepspeed.utils.safe_get_full_grad` to inspect full gradients before step (when DeepSpeed installed).
"""),

    code("""# DeepSpeed ZeRO NaN debugging patterns (fallback when DeepSpeed not installed)
try:
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    HAS_DS = True
except ImportError:
    HAS_DS = False

def inspect_grads_zero_style(model, rank=0):
    \"\"\"ZeRO-style: gather and check full grad for NaN.\"\"\"
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if HAS_DS:
            try:
                g = safe_get_full_grad(p)
            except Exception:
                pass
        if torch.isnan(g).any() or torch.isinf(g).any():
            bad.append(name)
    if bad and rank == 0:
        print(f"NaN/Inf in {len(bad)} params: {bad[:3]}...")
    return bad

model = nn.Linear(10, 2)
model.weight.grad = torch.tensor([[1.0, float('nan')]])
bad = inspect_grads_zero_style(model)
print(f"DeepSpeed available: {HAS_DS}, bad params found: {len(bad)}")

print('ZeRO debugging checklist:')
print('  1. Enable grad clipping in DeepSpeed config: gradient_clipping=1.0')
print('  2. Use fp16/bf16 config with dynamic loss scaling (fp16) or bf16')
print('  3. Inspect with safe_get_full_grad before optimizer step')
print('  4. Stage 3: check GatheredParameters gather timing')
"""),

    md("""## 12. FSDP Mixed Precision — Safe vs Unsafe Combinations

FSDP casts parameters for forward compute. Wrong dtype combo → overflow in forward, NaN in backward.

**Recommended (safe):**

```python
from torch.distributed.fsdp import MixedPrecision
mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
```

| param_dtype | reduce_dtype | buffer_dtype | Verdict |
|-------------|--------------|--------------|---------|
| bfloat16 | float32 | float32 | Safe (recommended) |
| float16 | float32 | float32 | OK with GradScaler |
| float16 | float16 | float16 | Risky — NaN prone |
| bfloat16 | bfloat16 | float16 | Avoid |
"""),

    code("""# FSDP MixedPrecision config examples (import-only demo on CPU)
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    SAFE_MP = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    RISKY_MP = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    print("Safe FSDP MixedPrecision:", SAFE_MP)
    print("Risky FSDP MixedPrecision:", RISKY_MP)
except ImportError:
    print("FSDP requires PyTorch 2.x — configs shown in markdown table above")

configs = [
    ("bf16/fp32/fp32", "Safe"),
    ("fp16/fp32/fp32", "OK with scaler"),
    ("fp16/fp16/fp16", "NaN prone"),
    ("bf16/bf16/fp16", "Avoid"),
]
print("\\n".join(f"  {c[0]:20s} -> {c[1]}" for c in configs))
"""),

    md("""## 13. Gradient Checkpointing + FSDP + AMP — Wrapper Order

Each technique alone is stable; **combining all three** requires correct nesting:

```
CORRECT (outermost → innermost):
  FSDP(
    model with checkpointed blocks,
    mixed_precision=MixedPrecision(...),
  )
  + autocast in training loop
  + GradScaler only for fp16 (not bf16)

WRONG: checkpoint inside FSDP without use_reentrant=False (PyTorch 2.1+)
WRONG: autocast wrapping FSDP module incorrectly
```

Order matters because checkpoint recomputation must see the same dtype context as forward.
"""),

    code("""# Correct wrapping order skeleton
WRAPPING_GUIDE = '''
# 1. Build model with gradient checkpointing on blocks
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
    def forward(self, x):
        return F.relu(self.lin(x))

class Model(nn.Module):
    def __init__(self, d=64, n=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False)  # PyTorch 2.1+
        return x

# 2. Wrap with FSDP (outermost)
# model = FSDP(Model(), mixed_precision=SAFE_MP, device_id=rank)

# 3. Training loop: autocast + scaler outside forward
# with autocast(dtype=torch.bfloat16):
#     loss = model(batch)
# scaler.scale(loss).backward()  # fp16 only
'''
print(WRAPPING_GUIDE)
"""),

    md("""## 14. DDP + AMP Full Training Loop (Multi-GPU Safe Pattern)

Production pattern combining: linear LR scaling, warmup, grad accumulation, sentinel, clipping.
"""),

    code("""def ddp_safe_step(model, optimizer, scaler, batch, accum_steps, step_in_accum,
                    sentinel, max_norm=1.0, use_amp=True):
    \"\"\"One micro-step of DDP-safe AMP training.\"\"\"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
        loss = model(batch).mean() / accum_steps

    if torch.isnan(loss) or torch.isinf(loss):
        return None, True

    scaler.scale(loss).backward()
    is_last = (step_in_accum + 1) % accum_steps == 0
    if is_last:
        scaler.unscale_(optimizer)
        sentinel.sanitize(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return loss.item() * accum_steps, False

print("Use with: DistributedDataParallel(model), DistributedSampler, linear LR scaling")
"""),

    md("""## 15. Distributed NaN Debugging Scenarios

Four real-world cases mirroring notebook 05 — with distributed-specific fixes.
"""),

    md("""### Scenario A: CLIP on 4 GPUs — NaN at Step 500

Same root cause as notebook 05 Scenario 1, plus **missing 4× LR scale** if global batch increased. Fix: clamp τ, BF16, grad clip, scale LR by GPU count.
"""),

    code("""# Distributed CLIP fix checklist
fixes = {
    "clamp_tau": "tau = log_tau.exp().clamp(min=0.01)",
    "bf16": "autocast(dtype=torch.bfloat16)",
    "grad_clip": "clip_grad_norm_(model.parameters(), 1.0)",
    "lr_scale": "lr = base_lr * num_gpus * local_batch / base_batch",
    "warmup": "warmup_steps = 500 * num_gpus",
}
for k, v in fixes.items():
    print(f"  {k:12s}: {v}")
"""),

    md("""### Scenario B: 8 GPUs, NaN After 10k Steps — Loss Scale = 1

Monitor `scaler.get_scale()` across ranks. When scale hits 1, FP16 is one overflow away from death. Migrate to BF16 + weight decay.
"""),

    code("""# Track loss scale across simulated 8-GPU run
scales = []
monitor = LossScaleMonitor(init_scale=65536.0)
grad_norms = list(np.linspace(0.5, 50, 200))  # creeping norms
for step, gn in enumerate(grad_norms):
    had_nan = gn > 40
    s = monitor.step(had_nan)
    scales.append(s)
    if s <= 1.0:
        print(f"Step {step}: loss scale hit {s:.1f} — migrate to BF16 NOW")
        break

plt.plot(scales); plt.title('Loss scale decay before late NaN'); plt.xlabel('Step'); plt.show()
"""),

    md("""### Scenario C: Rank 3 NaN First — Data Poisoning via AllReduce

Use `DistributedDataParallel` with `find_unused_parameters=False` only when safe. Always validate local batch before forward on **every** rank.
"""),

    code("""# Rank-aware batch validation before DDP forward
class DistributedDataGuard:
    def __init__(self, rank):
        self.rank = rank
        self.validator = DataValidator(rank)

    def safe_forward(self, model, batch):
        if not self.validator.validate(batch):
            return None
        out = model(batch)
        if torch.isnan(out).any():
            print(f"[rank {self.rank}] NaN in forward output — skip backward")
            return None
        return out

# Reuse DataValidator from notebook 05 toolkit pattern
class DataValidator:
    def __init__(self, rank=0):
        self.rank = rank
        self.skipped = 0
    def validate(self, batch):
        if torch.isnan(batch).any() or torch.isinf(batch).any():
            self.skipped += 1
            return False
        return True

guard = DistributedDataGuard(rank=3)
batch = torch.tensor([[1.0, float('nan')]])
result = guard.safe_forward(nn.Linear(2, 2), batch)
print(f"Corrupted batch blocked: {result is None}")
"""),

    md("""### Scenario D: LLaVA + LoRA + FSDP on 4 GPUs

Safe config table for multimodal finetuning without projector NaN.
"""),

    code("""LLAVA_FSDP_SAFE = {
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_scaling": "alpha/r = 1.0",
    "precision": "bfloat16",
    "fsdp_mp": "MixedPrecision(bf16, fp32 reduce, fp32 buffer)",
    "grad_clip": 1.0,
    "lr": "2e-5 * num_gpus (with warmup)",
    "warmup_ratio": 0.03,
    "checkpoint": "use_reentrant=False on projector blocks only",
}
for k, v in LLAVA_FSDP_SAFE.items():
    print(f"{k:18s}: {v}")
"""),

    md("""## 16. LARS / LAMB — When Linear Scaling Breaks

For batch sizes above ~8k (common at 64+ GPUs), plain SGD + linear scaling diverges.

**LARS** (You et al. 2017): layer-wise adaptive scaling for large batch.

**LAMB** (You et al. 2020): Adam-like trust ratio, used in BERT pretraining at batch 64k.

Switch when: NaN persists despite correct LR scaling + warmup + BF16.
"""),

    code("""# When to switch from linear scaling to LARS/LAMB
global_batches = [128, 512, 2048, 8192, 32768]
for gb in global_batches:
    rec = "Linear scaling + warmup" if gb < 8192 else "LARS/LAMB required"
    print(f"global_batch={gb:5d} -> {rec}")
"""),

    md("""## 17. Megatron-LM & 3D Parallelism NaN Notes

Tensor + pipeline + data parallel (Shoeybi et al. 2019, Narayanan et al. 2021):

- Tensor parallel splits attention softmax across ranks → need **vocab-parallel cross-entropy** (stable log-sum-exp)
- Pipeline bubbles don't cause NaN but mask **which stage** produced NaN — log micro-batch ID + stage rank
- Always use **loss scaling** or BF16 in Megatron-style training
"""),

    md("""## 18. Multi-GPU NaN Debugging Checklist (Distributed)

```
Distributed NaN?
├── Scaled LR with GPU count? (eta * global_batch / base_batch)
├── Warmup scaled with G?
├── FSDP MixedPrecision safe combo?
├── Grad checkpoint use_reentrant=False?
├── ZeRO stage: safe_get_full_grad check?
├── All ranks same code path? (no rank-0-only collectives)
├── NCCL_DEBUG=INFO for hangs
└── Per-rank grad norm logging before AllReduce
```
"""),

    md("""## 19. Environment Variables for Multi-GPU Debugging

```bash
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1   # slow but pinpoints NaN kernel
```
"""),

    code("""# Launch helper (print recommended env vars)
import os
DEBUG_ENV = {
    "NCCL_DEBUG": "INFO",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
}
print("Recommended debug env (set before torchrun):")
for k, v in DEBUG_ENV.items():
    print(f"  export {k}={v}")
"""),

    md("""## 20. Updated Scaling Recipe Table (with NaN Prevention)

| GPUs | Recipe | NaN Prevention |
|------|--------|----------------|
| 1 | AMP + checkpointing | BF16, grad clip |
| 2–8 | DDP + linear LR scale | Warmup ∝ G, sentinel |
| 2–8 large model | FSDP + safe MixedPrecision | bf16/fp32/fp32 |
| 8–64 | FSDP + tensor parallel | LARS if batch > 8k |
| 64+ | 3D parallel + DeepSpeed | ZeRO grad inspect, stable CE |
"""),
]

NB06_REFS = """## References & Further Reading

### Papers
- Micikevicius et al. (2018) — Mixed Precision Training — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Ott et al. (2019) — fairseq: FP16 Training — [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Rajbhandari et al. (2020) — ZeRO: Memory Optimizations — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Shoeybi et al. (2019) — Megatron-LM — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Goyal et al. (2017) — Large Minibatch SGD — [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- You et al. (2017) — LARS — [arXiv:1708.03888](https://arxiv.org/abs/1708.03888)
- You et al. (2020) — LAMB — [arXiv:1904.00962](https://arxiv.org/abs/1904.00962)
- Narayanan et al. (2021) — Megatron-LM Pipeline — [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- Li et al. (2020) — PyTorch Distributed — [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Zhao et al. (2023) — PyTorch FSDP — [Blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

### Blogs & Docs
- [Lilian Weng — Training Large Neural Networks](https://lilianweng.github.io/posts/2021-09-25-train-compute/)
- [HuggingFace — Debugging Mixed Precision](https://huggingface.co/docs/transformers/perf_train_gpu_one#mixed-precision-training)
- [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA — Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
"""


if __name__ == "__main__":
    insert_before_references(NB05, NB05_NEW, NB05_REFS)
    insert_before_references(NB06, NB06_NEW, NB06_REFS)
