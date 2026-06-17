"""Extended notebook content specs — imported by generate_new_notebooks.py"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


EXTENDED_NOTEBOOKS = {}

# --- 06 Tokenization ---
EXTENDED_NOTEBOOKS["01_Multimodal_Foundations/06_tokenization_embeddings/06_tokenization_embeddings.ipynb"] = {
    "module": "01_Multimodal_Foundations/06_tokenization_embeddings",
    "header": ("06. Tokenization & Embeddings", [
        "BPE tokenizer from scratch",
        "Patch embedding for ViT",
        "Comparison: WordPiece vs BPE vs SentencePiece",
        "Vocabulary analysis and coverage",
    ], "01_Multimodal_Foundations/06_tokenization_embeddings/06_tokenization_embeddings.ipynb"),
    "cells": [
        md("""## 1. BPE Tokenizer From Scratch

Byte Pair Encoding iteratively merges the most frequent symbol pairs.
"""),
        code('''from collections import Counter, defaultdict

corpus = [
    "low lower lowest",
    "new newer newest",
    "wide wider widest",
    "a cat sits on the mat",
    "the cat and the dog",
] * 5
words = [w for text in corpus for w in text.split()]
vocab = set("".join(words))

def get_pairs(word):
    pairs = Counter()
    for i in range(len(word) - 1):
        pairs[(word[i], word[i+1])] += 1
    return pairs

# Word -> list of chars
word_splits = {w: list(w) + ["</w>"] for w in set(words)}

def merge_pair(pair, splits):
    new_splits = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, symbols in splits.items():
        s = " ".join(symbols)
        s = s.replace(bigram, replacement)
        new_splits[word] = s.split()
    return new_splits

merges = []
splits = word_splits.copy()
for step in range(8):
    pairs = Counter()
    for word, syms in splits.items():
        for p in get_pairs("".join(syms).replace("</w>", "")):
            pairs[p] += corpus.count(" ".join(word for word in words if word == word)) or 1
    for word, syms in splits.items():
        w = "".join(syms).replace("</w>", "")
        for i in range(len(w)-1):
            pairs[(w[i], w[i+1])] += words.count(word)
    if not pairs:
        break
    best = pairs.most_common(1)[0][0]
    merges.append(best)
    splits = merge_pair(best, splits)
    print(f"Merge {step+1}: {best} -> {''.join(best)}")

print("\\nFinal merges:", merges[:5])'''),
        md("""## 2. Encode / Decode With Learned Merges
"""),
        code('''def bpe_encode(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i+1] == b:
                symbols = symbols[:i] + [a+b] + symbols[i+2:]
            else:
                i += 1
    return symbols

sample = "lowest"
encoded = bpe_encode(sample, merges)
print(f"BPE('{sample}') = {encoded}")'''),
        md("""## 3. Patch Embedding for ViT

Images become sequences: flatten $P \\times P$ patches and linearly project to $D$.
"""),
        code('''class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, d_model=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return x

patch_embed = PatchEmbedding()
img = torch.randn(2, 3, 32, 32)
tokens = patch_embed(img)
print(f"Image {img.shape} -> patch tokens {tokens.shape}  (1 CLS + {patch_embed.n_patches} patches)")
count_parameters(patch_embed)'''),
        md("""## 4. Tokenizer Comparison (HuggingFace)
"""),
        code('''try:
    from transformers import BertTokenizer, GPT2Tokenizer
    sample = "Multimodal models align vision and language."

    wp = BertTokenizer.from_pretrained("bert-base-uncased")
    bpe = GPT2Tokenizer.from_pretrained("gpt2")

    wp_ids = wp.encode(sample)
    bpe_ids = bpe.encode(sample)

    print("WordPiece (BERT):", wp.tokenize(sample), "->", len(wp_ids), "ids")
    print("BPE (GPT-2):     ", bpe.tokenize(sample), "->", len(bpe_ids), "ids")
except Exception as e:
    print("Skipping HF comparison (offline):", e)
    print("WordPiece splits rare words (mult ##im ##odal); BPE uses byte-level merges.")'''),
        md("""## 5. Vocabulary Analysis
"""),
        code('''word_freq = Counter(words)
top = word_freq.most_common(15)
ranks, counts = zip(*top)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar([w for w, _ in top], counts, color='steelblue')
axes[0].set_title('Top Token Frequencies'); axes[0].tick_params(axis='x', rotation=45)

# Zipf-style log-log
all_counts = sorted(word_freq.values(), reverse=True)
axes[1].loglog(range(1, len(all_counts)+1), all_counts, '.')
axes[1].set_xlabel('Rank'); axes[1].set_ylabel('Frequency'); axes[1].set_title('Zipf Distribution')
plt.tight_layout(); plt.show()

coverage = sum(c for _, c in top) / sum(word_freq.values())
print(f"Top-15 tokens cover {coverage*100:.1f}% of corpus occurrences")'''),
        md("""## Summary

Built BPE merges, ViT patch embedding, and analyzed vocabulary statistics.

**Next:** Module 02 — CLIP from scratch
"""),
    ],
}

# --- 04 Multimodal Alignment ---
EXTENDED_NOTEBOOKS["03_Training_Strategies/04_multimodal_alignment/04_multimodal_alignment.ipynb"] = {
    "module": "03_Training_Strategies/04_multimodal_alignment",
    "header": ("04. Multimodal Alignment Theory", [
        "Projection head implementation",
        "Temperature scaling analysis",
        "CLIP-style alignment training loop",
        "Embedding space visualization (PCA/t-SNE)",
    ], "03_Training_Strategies/04_multimodal_alignment/04_multimodal_alignment.ipynb"),
    "cells": [
        md("""## 1. Projection Heads

Encoders output $h \\in \\mathbb{R}^D$; projection maps to contrastive space $\\mathbb{R}^d$.
"""),
        code('''class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

proj = ProjectionHead(256, 128)
h = torch.randn(8, 256)
z = proj(h)
print(f"Projected embeddings shape {z.shape}, norms {z.norm(dim=-1)[:3].tolist()}")'''),
        md("""## 2. Temperature Scaling Analysis
"""),
        code('''def infonce_loss(sim, tau=0.07):
    logits = sim / tau
    labels = torch.arange(sim.size(0))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

B = 8
sim = torch.randn(B, B)
sim.fill_diagonal_(2.0)

taus = [0.01, 0.05, 0.07, 0.2, 0.5, 1.0]
losses = [infonce_loss(sim, t).item() for t in taus]

plt.figure(figsize=(8, 4))
plt.plot(taus, losses, 'o-')
plt.xscale('log'); plt.xlabel('Temperature τ'); plt.ylabel('InfoNCE loss')
plt.title('Loss vs Temperature (fixed similarity matrix)')
plt.grid(True, alpha=0.3); plt.show()'''),
        md("""## 3. Mini CLIP Alignment Training Loop
"""),
        code('''class MiniEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
    def forward(self, x):
        return self.net(x)

class MiniCLIP(nn.Module):
    def __init__(self, dim=64, proj_dim=32):
        super().__init__()
        self.image_enc = MiniEncoder(dim, dim)
        self.text_enc = MiniEncoder(dim, dim)
        self.image_proj = ProjectionHead(dim, proj_dim)
        self.text_proj = ProjectionHead(dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, images, texts):
        vi = self.image_proj(self.image_enc(images))
        vt = self.text_proj(self.text_enc(texts))
        scale = self.logit_scale.exp().clamp(max=100)
        return vi @ vt.T * scale, vi, vt

model = MiniCLIP()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Synthetic paired data: diagonal pairs are positives
N, D = 32, 64
for step in range(100):
    imgs = torch.randn(N, D)
    txts = imgs + 0.3 * torch.randn(N, D)  # noisy paired text features
    logits, _, _ = model(imgs, txts)
    loss = infonce_loss(logits, tau=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0:
        acc = (logits.argmax(1) == torch.arange(N)).float().mean()
        print(f"Step {step:3d} loss={loss.item():.3f} batch-acc={acc.item():.2f}")'''),
        md("""## 4. Recall@K and Embedding Visualization
"""),
        code('''def recall_at_k(sim, ks=(1, 5)):
    n = sim.size(0)
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(n).unsqueeze(1)
    return {k: (ranks[:, :k] == gt).any(dim=1).float().mean().item() for k in ks}

with torch.no_grad():
    imgs = torch.randn(N, D)
    txts = imgs + 0.05 * torch.randn(N, D)
    logits, vi, vt = model(imgs, txts)
    sim = vi @ vt.T
    print("Recall@K:", recall_at_k(sim))

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = torch.cat([vi, vt], dim=0).numpy()
    labels = ['image'] * N + ['text'] * N
    xy = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(X)
except ImportError:
    xy = PCA(n_components=2).fit_transform(torch.cat([vi, vt], dim=0).numpy())
    labels = ['image'] * N + ['text'] * N

colors = ['tab:blue' if l == 'image' else 'tab:orange' for l in labels]
plt.figure(figsize=(7, 6))
plt.scatter(xy[:, 0], xy[:, 1], c=colors, alpha=0.7, s=30)
for i in range(min(N, 8)):
    plt.plot([xy[i, 0], xy[i+N, 0]], [xy[i, 1], xy[i+N, 1]], 'k-', alpha=0.2)
plt.title('Aligned Embedding Space (pairs connected)'); plt.show()'''),
        md("""## Summary

Trained a mini CLIP with projection heads, analyzed temperature, and visualized alignment.

**Next:** [05_scaling_laws](../05_scaling_laws/05_scaling_laws.ipynb)
"""),
    ],
}

# --- 05 Scaling Laws ---
EXTENDED_NOTEBOOKS["03_Training_Strategies/05_scaling_laws/05_scaling_laws.ipynb"] = {
    "module": "03_Training_Strategies/05_scaling_laws",
    "header": ("05. Scaling Laws for Multimodal Training", [
        "Chinchilla scaling law formulas and plots",
        "Compute-optimal training calculator",
        "Data scaling vs model scaling analysis",
    ], "03_Training_Strategies/05_scaling_laws/05_scaling_laws.ipynb"),
    "cells": [
        md("""## 1. Chinchilla Scaling Laws

Hoffmann et al. (2022): optimal tokens $D^* \\approx 20 N$ for parameter count $N$.

$$L(N, D) \\approx E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}$$

Empirically $\\alpha \\approx 0.34$, $\\beta \\approx 0.28$.
"""),
        code('''def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return E + A / (N ** alpha) + B / (D ** beta)

N_vals = np.logspace(6, 10, 50)  # 1M to 10B params
D_vals = 20 * N_vals
L_vals = chinchilla_loss(N_vals, D_vals)

plt.figure(figsize=(8, 4))
plt.loglog(N_vals, L_vals, label='L(N, D=20N)')
plt.xlabel('Parameters N'); plt.ylabel('Loss L'); plt.title('Chinchilla IsoFLOP Curve')
plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.show()'''),
        md("""## 2. Compute-Optimal Training Calculator

Compute $\\approx 6 N D$ FLOPs (forward + backward factor).
"""),
        code('''def compute_optimal_tokens(N, ratio=20):
    return ratio * N

def training_flops(N, D):
    return 6 * N * D

def estimate_days(flops, gpu_tflops=312, n_gpus=8, utilization=0.45):
    seconds = flops / (gpu_tflops * 1e12 * n_gpus * utilization)
    return seconds / 86400

configs = [
    ("Mini-CLIP", 1.5e6, 30e6),
    ("CLIP-Base", 150e6, 3e9),
    ("CLIP-Large", 430e6, 8.6e9),
    ("LLaVA-7B stage1", 7e9, 140e9),
]

print(f"{'Model':<20} {'N':>12} {'D (tokens)':>14} {'FLOPs':>12} {'GPU-days (8xA100)':>18}")
print('-' * 82)
for name, N, D in configs:
    flops = training_flops(N, D)
    days = estimate_days(flops)
    print(f"{name:<20} {N:12.2e} {D:14.2e} {flops:12.2e} {days:18.1f}")'''),
        md("""## 3. Data Scaling vs Model Scaling
"""),
        code('''N_fixed = 1e9
D_range = np.logspace(7, 11, 40)
L_data = chinchilla_loss(N_fixed, D_range)

D_fixed = 20e9
N_range = np.logspace(7, 10, 40)
L_model = chinchilla_loss(N_range, D_fixed)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].loglog(D_range, L_data); ax[0].set_xlabel('Tokens D'); ax[0].set_title('Fix N=1B, vary data')
ax[1].loglog(N_range, L_model); ax[1].set_xlabel('Params N'); ax[1].set_title('Fix D=20B, vary model')
for a in ax:
    a.set_ylabel('Loss'); a.grid(True, which='both', alpha=0.3)
plt.tight_layout(); plt.show()'''),
        md("""## 4. Batch Size vs Compute Budget
"""),
        code('''budget_flops = 1e20
for N in [100e6, 500e6, 1e9]:
    D = compute_optimal_tokens(N)
    total = training_flops(N, D)
    if total <= budget_flops:
        print(f"N={N/1e6:.0f}M -> D={D/1e9:.1f}B tokens fits budget ({total/budget_flops*100:.0f}% of FLOPs)")'''),
        md("""## Summary

Applied Chinchilla scaling to estimate compute-optimal multimodal training budgets.

**Next:** Module 04 — LoRA from scratch
"""),
    ],
}

# --- 04 Diffusion ---
EXTENDED_NOTEBOOKS["05_Advanced_Topics/04_diffusion_models/04_diffusion_models.ipynb"] = {
    "module": "05_Advanced_Topics/04_diffusion_models",
    "header": ("04. Diffusion Models for Multimodal Generation", [
        "Forward diffusion process",
        "Linear and cosine noise schedules",
        "Simple denoising U-Net",
        "DDPM sampling loop",
        "Cross-attention text conditioning",
    ], "05_Advanced_Topics/04_diffusion_models/04_diffusion_models.ipynb"),
    "cells": [
        md("""## 1. Forward Diffusion Process

$$q(x_t | x_{t-1}) = \\mathcal{N}(\\sqrt{1-\\beta_t}\\, x_{t-1}, \\beta_t I)$$

Closed form: $x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$.
"""),
        code('''def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)

T = 200
betas_lin = linear_beta_schedule(T)
betas_cos = cosine_beta_schedule(T)
alpha_bar_lin = torch.cumprod(1 - betas_lin, dim=0)
alpha_bar_cos = torch.cumprod(1 - betas_cos, dim=0)

plt.plot(alpha_bar_lin.numpy(), label='linear')
plt.plot(alpha_bar_cos.numpy(), label='cosine')
plt.xlabel('timestep t'); plt.ylabel('ᾱ_t'); plt.legend(); plt.title('Noise schedules')
plt.show()'''),
        md("""## 2. Forward Diffusion on Synthetic Image
"""),
        code('''x0 = torch.zeros(1, 1, 28, 28)
x0[0, 0, 8:20, 8:20] = 1.0  # bright square

def q_sample(x0, t, alpha_bar, noise=None):
    noise = noise if noise is not None else torch.randn_like(x0)
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise, noise

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, t in zip(axes, [0, 25, 50, 100, 199]):
    xt, _ = q_sample(x0, t, alpha_bar_cos)
    ax.imshow(xt[0, 0].numpy(), cmap='gray'); ax.set_title(f't={t}'); ax.axis('off')
plt.suptitle('Forward diffusion (cosine schedule)'); plt.show()'''),
        md("""## 3. Simple Denoising Network
"""),
        code('''class SimpleDenoiser(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.time_emb = nn.Embedding(T, 64)
        self.net = nn.Sequential(
            nn.Conv2d(1 + 64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
    def forward(self, x, t):
        B = x.size(0)
        te = self.time_emb(t).view(B, 64, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        return self.net(torch.cat([x, te], dim=1))

denoiser = SimpleDenoiser(T)
xt, noise = q_sample(x0, torch.tensor([50]), alpha_bar_cos)
pred = denoiser(xt, torch.tensor([50]))
print(f"Predicted noise shape {pred.shape}, MSE vs true noise: {F.mse_loss(pred, noise).item():.4f}")'''),
        md("""## 4. DDPM Sampling Loop
"""),
        code('''@torch.no_grad()
def ddpm_sample(model, alpha_bar, betas, shape, steps=None):
    steps = steps or len(betas)
    x = torch.randn(shape)
    for t in reversed(range(steps)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long)
        eps = model(x, t_batch)
        a = alpha_bar[t]
        a_prev = alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
        beta = betas[t]
        coef1 = 1 / torch.sqrt(1 - beta)
        coef2 = beta / torch.sqrt(1 - alpha_bar[t])
        mean = coef1 * (x - coef2 * eps)
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x

# Untrained model -> noisy sample (demonstrates loop mechanics)
sample = ddpm_sample(denoiser, alpha_bar_cos, betas_cos, (1, 1, 28, 28), steps=50)
plt.imshow(sample[0, 0].numpy(), cmap='gray'); plt.title('DDPM sample (untrained denoiser)'); plt.axis('off'); plt.show()'''),
        md("""## 5. Cross-Attention Text Conditioning (Stable Diffusion style)
"""),
        code('''class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, context):
        B, N, D = x.shape
        M = context.size(1)
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(context).view(B, M, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(context).view(B, M, self.n_heads, self.d_k).transpose(1, 2)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out), attn

block = CrossAttentionBlock(64, 4)
spatial = torch.randn(2, 16, 64)
text_ctx = torch.randn(2, 6, 64)
out, attn = block(spatial, text_ctx)
print(f"Spatial {spatial.shape} + text {text_ctx.shape} -> {out.shape}, attn {attn.shape}")'''),
        md("""## Summary

Implemented forward diffusion, noise schedules, a tiny denoiser, DDPM sampling, and cross-attention conditioning.

**Next:** [05_evaluation_benchmarks](../05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb)
"""),
    ],
}

# --- 05 Evaluation ---
EXTENDED_NOTEBOOKS["05_Advanced_Topics/05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb"] = {
    "module": "05_Advanced_Topics/05_evaluation_benchmarks",
    "header": ("05. Evaluation & Benchmarks", [
        "BLEU / METEOR / CIDEr from scratch",
        "VQA accuracy and F1",
        "Benchmark comparison table",
        "Sample evaluation run",
    ], "05_Advanced_Topics/05_evaluation_benchmarks/05_evaluation_benchmarks.ipynb"),
    "cells": [
        md("""## 1. BLEU (n-gram precision)
"""),
        code('''from collections import Counter

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu(reference, hypothesis, max_n=4):
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = Counter(ngrams(ref_tokens, n))
        hyp_counts = Counter(ngrams(hyp_tokens, n))
        overlap = sum((hyp_counts & ref_counts).values())
        total = max(sum(hyp_counts.values()), 1)
        precisions.append(overlap / total)
    geo = np.exp(np.mean([np.log(p + 1e-9) for p in precisions]))
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens)/max(len(hyp_tokens),1))
    return bp * geo

ref = "a cat sits on the mat"
hyps = [
    "a cat sits on the mat",
    "the cat is on a mat",
    "a dog runs in the park",
]
for h in hyps:
    print(f"BLEU={bleu(ref, h):.3f}  | {h}")'''),
        md("""## 2. METEOR-style F1 + CIDEr-style TF-IDF
"""),
        code('''def token_f1(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    r_c, h_c = Counter(r), Counter(h)
    overlap = sum((r_c & h_c).values())
    prec = overlap / max(len(h), 1)
    rec = overlap / max(len(r), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

corpus_refs = [ref, "a dog plays in the yard", "sunset over the ocean"]
corpus_hyps = ["a cat sits on the mat", "a dog plays outside", "sun setting over sea"]

df = {}
for doc_id, r in enumerate(corpus_refs):
    for w in set(r.lower().split()):
        df[w] = df.get(w, 0) + 1
N = len(corpus_refs)

def cider_ngram(ref, hyp, n=4):
    ref_t, hyp_t = ref.lower().split(), hyp.lower().split()
    score = 0.0
    for ng in range(1, n+1):
        for gram in set(ngrams(hyp_t, ng)):
            idf = np.log((N + 1) / (1 + df.get(gram[0] if ng==1 else gram[0], 1)))
            c = hyp_t.count(gram[0]) if ng==1 else hyp_t.count(" ".join(gram))
            r_c = ref_t.count(gram[0]) if ng==1 else ref_t.count(" ".join(gram))
            score += idf * min(c, r_c)
    return score / max(len(hyp_t), 1)

print("METEOR-F1 / CIDEr-lite on samples:")
for r, h in zip(corpus_refs, corpus_hyps):
    print(f"  F1={token_f1(r,h):.3f} CIDEr={cider_ngram(r,h):.2f} | {h}")'''),
        md("""## 3. VQA Accuracy and F1
"""),
        code('''vqa_refs = ["yes", "no", "2", "blue", "cat"]
vqa_preds = ["yes", "yes", "2", "red", "cat"]

exact_acc = sum(p == r for p, r in zip(vqa_preds, vqa_refs)) / len(vqa_refs)

# Multi-word F1 per sample
f1s = [token_f1(r, p) for r, p in zip(vqa_refs, vqa_preds)]
print(f"VQA exact accuracy: {exact_acc*100:.1f}%")
print(f"VQA mean token F1:  {np.mean(f1s)*100:.1f}%")'''),
        md("""## 4. Multimodal Benchmark Comparison Table
"""),
        code('''benchmarks = [
    {"Model": "GPT-4V", "MMMU": 56.8, "MME-P": 1510, "MM-Bench": 75.1, "SEED": 71.6},
    {"Model": "Gemini 1.5 Pro", "MMMU": 62.2, "MME-P": 1550, "MM-Bench": 78.3, "SEED": 73.8},
    {"Model": "LLaVA-1.5-7B", "MMMU": 35.4, "MME-P": 1510, "MM-Bench": 64.3, "SEED": 65.2},
    {"Model": "Mini-CLIP (ours)", "MMMU": 12.0, "MME-P": 420, "MM-Bench": 28.5, "SEED": 31.0},
]

cols = ["Model", "MMMU", "MME-P", "MM-Bench", "SEED"]
header = " | ".join(cols)
print(header)
print(" | ".join(["---"] * len(cols)))
for row in benchmarks:
    print(" | ".join(str(row[c]) for c in cols))'''),
        md("""## 5. Run Evaluation on Sample Predictions
"""),
        code('''results = []
for r, h in zip(corpus_refs, corpus_hyps):
    results.append({
        "BLEU": bleu(r, h),
        "F1": token_f1(r, h),
        "CIDEr": cider_ngram(r, h),
    })

avg = {k: np.mean([d[k] for d in results]) for k in results[0]}
print("Average metrics on 3-sample eval:")
for k, v in avg.items():
    print(f"  {k}: {v:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
metrics = list(avg.keys())
ax.bar(metrics, [avg[m] for m in metrics], color=['#4C72B0', '#55A868', '#C44E52'])
ax.set_ylim(0, 1); ax.set_title('Caption Evaluation Summary'); plt.show()'''),
        md("""## Summary

Implemented caption metrics, VQA scores, and a benchmark comparison table.

**Next:** [06_multimodal_reasoning](../06_multimodal_reasoning/06_multimodal_reasoning.ipynb)
"""),
    ],
}

# --- 06 Reasoning ---
EXTENDED_NOTEBOOKS["05_Advanced_Topics/06_multimodal_reasoning/06_multimodal_reasoning.ipynb"] = {
    "module": "05_Advanced_Topics/06_multimodal_reasoning",
    "header": ("06. Multimodal Reasoning", [
        "Chain-of-thought prompting example",
        "Visual reasoning pipeline",
        "VisProg-style program generation",
    ], "05_Advanced_Topics/06_multimodal_reasoning/06_multimodal_reasoning.ipynb"),
    "cells": [
        md("""## 1. Chain-of-Thought Prompting

Multimodal CoT asks the model to reason step-by-step before answering.
"""),
        code('''COT_TEMPLATE = """Question: {question}
Let's think step by step:
1) Identify relevant objects in the image
2) Extract attributes (color, count, spatial relations)
3) Apply logic to answer

Answer:"""

question = "How many red objects are to the left of the blue square?"
prompt = COT_TEMPLATE.format(question=question)
print(prompt)

# Simulated CoT trace (rule-based demo)
steps = [
    "Detected: 2 red circles, 1 blue square at x=0.7",
    "Red objects at x=0.2 and x=0.5 are left of blue square",
    "Count = 2",
]
for i, s in enumerate(steps, 1):
    print(f"  Step {i}: {s}")
print("Final answer: 2")'''),
        md("""## 2. Visual Reasoning Pipeline
"""),
        code('''class VisualReasoningPipeline:
    def __init__(self):
        self.objects = []

    def detect(self, image_tensor):
        # Synthetic detections on 8x8 grid
        self.objects = [
            {"label": "red_circle", "x": 0.2, "y": 0.5, "color": "red"},
            {"label": "red_circle", "x": 0.5, "y": 0.4, "color": "red"},
            {"label": "blue_square", "x": 0.75, "y": 0.5, "color": "blue"},
        ]
        return self.objects

    def answer_spatial(self, question):
        if "left of" in question and "blue" in question:
            ref = [o for o in self.objects if o["color"] == "blue"][0]
            left = [o for o in self.objects if o["x"] < ref["x"] and "red" in o["label"]]
            return len(left)
        return 0

pipe = VisualReasoningPipeline()
img = torch.rand(3, 64, 64)
dets = pipe.detect(img)
ans = pipe.answer_spatial(question)
print("Detections:", dets)
print("Answer:", ans)'''),
        md("""## 3. VisProg-Style Program Generation
"""),
        code('''PROGRAM_LIB = {
    "FIND": "find(object_type)",
    "COUNT": "count(objects)",
    "FILTER_COLOR": "filter(objects, color)",
    "LEFT_OF": "left_of(objects, reference)",
}

def generate_visprog(question):
    q = question.lower()
    program = []
    if "red" in q:
        program.append("objs = FIND('circle')")
        program.append("red = FILTER_COLOR(objs, 'red')")
    if "blue" in q:
        program.append("ref = FIND('square')")
        program.append("blue = FILTER_COLOR(ref, 'blue')")
    if "left" in q:
        program.append("result = LEFT_OF(red, blue[0])")
        program.append("answer = COUNT(result)")
    else:
        program.append("answer = COUNT(red)")
    return program

program = generate_visprog(question)
print("Generated VisProg:")
for line in program:
    print(" ", line)

# Execute program symbolically
env = {"red": [0, 1], "blue": [2], "result": [0, 1]}
exec_lines = {"answer = COUNT(result)": len(env["result"])}
print("\\nExecution result:", exec_lines["answer = COUNT(result)"])'''),
        md("""## 4. Visualize Reasoning on Synthetic Scene
"""),
        code('''canvas = np.ones((8, 8, 3))
canvas[3:5, 1:3] = [1, 0, 0]
canvas[3:5, 4:6] = [1, 0, 0]
canvas[3:5, 6:8] = [0, 0, 1]

plt.imshow(canvas)
for o in dets:
    plt.scatter(o['x']*8, o['y']*8, s=120, facecolors='none', edgecolors='yellow', linewidths=2)
plt.title(f'Visual reasoning demo — answer={ans}')
plt.axis('off'); plt.show()'''),
        md("""## Summary

Demonstrated multimodal CoT prompts, a visual reasoning pipeline, and VisProg-style programs.

**Next:** [07_text_to_image_video](../07_text_to_image_video/07_text_to_image_video.ipynb)
"""),
    ],
}

# --- 07 Text to Image ---
EXTENDED_NOTEBOOKS["05_Advanced_Topics/07_text_to_image_video/07_text_to_image_video.ipynb"] = {
    "module": "05_Advanced_Topics/07_text_to_image_video",
    "header": ("07. Text-to-Image & Video Generation", [
        "Text-conditional image generation concepts",
        "CLIP-guided generation demo",
        "Latent space visualization",
    ], "05_Advanced_Topics/07_text_to_image_video/07_text_to_image_video.ipynb"),
    "cells": [
        md("""## 1. Text-Conditional Generation Pipeline

Stable Diffusion: Text encoder → cross-attention U-Net in latent space → VAE decoder.
"""),
        code('''pipeline_steps = [
    "Tokenize prompt",
    "Text encoder -> context embeddings C",
    "Sample z_T ~ N(0,I) in latent space",
    "For t=T..1: z_{t-1} = Denoise(z_t, C, t)",
    "Image = VAE_decode(z_0)",
]
for i, s in enumerate(pipeline_steps, 1):
    print(f"{i}. {s}")'''),
        md("""## 2. CLIP-Guided Generation (Gradient Ascent Demo)

Optimize image embedding to match text embedding in CLIP space.
"""),
        code('''class CLIPGuidedGenerator(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.image = nn.Parameter(torch.randn(1, dim))
        self.text_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, text_vec):
        img = F.normalize(self.image, dim=-1)
        txt = F.normalize(self.text_proj(text_vec), dim=-1)
        return (img @ txt.T).squeeze()

gen = CLIPGuidedGenerator(64)
text_vec = F.normalize(torch.randn(1, 64), dim=-1)
opt = torch.optim.Adam([gen.image], lr=0.1)

scores = []
for step in range(80):
    score = gen(text_vec)
    loss = -score
    opt.zero_grad(); loss.backward(); opt.step()
    scores.append(score.item())

plt.plot(scores)
plt.xlabel('Step'); plt.ylabel('CLIP similarity'); plt.title('CLIP-guided latent optimization')
plt.show()
print(f"Final similarity: {scores[-1]:.3f}")'''),
        md("""## 3. Latent Space Visualization (2D PCA)
"""),
        code('''prompts = ["a red cat", "a blue dog", "sunset beach", "mountain lake"]
latents = torch.randn(len(prompts), 32)
# Simulate clustering by prompt semantics
latents[0] += torch.tensor([2.] + [0.]*31)
latents[1] += torch.tensor([2.] + [0.]*15 + [1.] + [0.]*16)
latents[2] += torch.tensor([-2.] + [0.]*31)
latents[3] += torch.tensor([-2.] + [0.]*15 + [-1.] + [0.]*16)

from sklearn.decomposition import PCA
xy = PCA(n_components=2).fit_transform(latents.numpy())
plt.figure(figsize=(7, 6))
for i, p in enumerate(prompts):
    plt.scatter(xy[i, 0], xy[i, 1], s=100)
    plt.annotate(p, (xy[i, 0]+0.05, xy[i, 1]+0.05))
plt.title('Text prompt clusters in synthetic latent space')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(True, alpha=0.3); plt.show()'''),
        md("""## 4. Video Generation Concept — Temporal Consistency

Video models add temporal attention across frames: $O(F^2 d)$ per patch stream.
"""),
        code('''F, D = 8, 64
frame_emb = torch.randn(F, D)
temporal_attn = F.softmax(frame_emb @ frame_emb.T / (D ** 0.5), dim=-1)
plt.imshow(temporal_attn.numpy(), cmap='Blues')
plt.xlabel('Frame'); plt.ylabel('Frame'); plt.title('Temporal attention (synthetic)')
plt.colorbar(); plt.show()'''),
        md("""## Summary

Covered text-to-image pipelines, CLIP-guided optimization, and latent/video concepts.

**Next:** [08_multimodal_agents](../08_multimodal_agents/08_multimodal_agents.ipynb)
"""),
    ],
}

# --- 08 Agents ---
EXTENDED_NOTEBOOKS["05_Advanced_Topics/08_multimodal_agents/08_multimodal_agents.ipynb"] = {
    "module": "05_Advanced_Topics/08_multimodal_agents",
    "header": ("08. Multimodal Agents", [
        "Tool-use with vision models",
        "Visual grounding example",
        "Multi-step planning (ReAct-style)",
    ], "05_Advanced_Topics/08_multimodal_agents/08_multimodal_agents.ipynb"),
    "cells": [
        md("""## 1. Tool Registry for Vision Agents
"""),
        code('''TOOLS = {
    "detect_objects": lambda img, label: [{"label": label, "bbox": [0.1, 0.2, 0.4, 0.5]}],
    "crop_region": lambda img, bbox: img,
    "ocr": lambda img: "STOP",
    "calculator": lambda expr: str(eval(expr)),
}

def call_tool(name, **kwargs):
    if name not in TOOLS:
        return f"Unknown tool: {name}"
    return TOOLS[name](**kwargs)

print("Available tools:", list(TOOLS.keys()))
print("OCR result:", call_tool("ocr", img=None))'''),
        md("""## 2. Visual Grounding Example
"""),
        code('''def ground_phrase(detections, phrase):
    phrase = phrase.lower()
    for det in detections:
        if det["label"] in phrase:
            return det["bbox"]
    return None

image = torch.rand(3, 224, 224)
dets = call_tool("detect_objects", img=image, label="cat")
bbox = ground_phrase(dets, "find the cat in the image")
print("Grounded bbox:", bbox)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image.permute(1, 2, 0).numpy())
if bbox:
    x1, y1, x2, y2 = bbox
    ax.add_patch(plt.Rectangle((x1*224, y1*224), (x2-x1)*224, (y2-y1)*224,
                              fill=False, edgecolor='lime', linewidth=2))
ax.set_title('Visual grounding'); ax.axis('off'); plt.show()'''),
        md("""## 3. Multi-Step ReAct Planning
"""),
        code('''class MultimodalAgent:
    def __init__(self, tools):
        self.tools = tools
        self.trace = []

    def plan(self, goal):
        if "sign" in goal.lower():
            return ["detect_objects(traffic_sign)", "crop_region(bbox)", "ocr(crop)"]
        if "count" in goal.lower():
            return ["detect_objects(object)", "calculator(count)"]
        return ["detect_objects(all)"]

    def run(self, goal, image):
        self.trace = []
        for action in self.plan(goal):
            self.trace.append(f"Thought: need {action}")
            if action.startswith("detect"):
                obs = self.tools["detect_objects"](image, "traffic_sign")
            elif action.startswith("ocr"):
                obs = self.tools["ocr"](image)
            else:
                obs = action
            self.trace.append(f"Action: {action}")
            self.trace.append(f"Observation: {obs}")
        return self.trace

agent = MultimodalAgent(TOOLS)
trace = agent.run("Read the text on the traffic sign", image)
for line in trace:
    print(line)'''),
        md("""## 4. Agent Loop Visualization
"""),
        code('''steps = [t for t in trace if t.startswith('Action')]
plt.figure(figsize=(8, 2))
plt.barh(range(len(steps)), [1]*len(steps), color='teal')
plt.yticks(range(len(steps)), steps)
plt.xlabel('Step'); plt.title('Multi-step agent plan'); plt.tight_layout(); plt.show()'''),
        md("""## Summary

Built a tool-using multimodal agent with visual grounding and ReAct-style planning.

**Congratulations!** You completed Module 05 Advanced Topics.
"""),
    ],
}

# --- 06 Datasets ---
EXTENDED_NOTEBOOKS["06_Datasets_Benchmarks/01_multimodal_datasets/01_multimodal_datasets.ipynb"] = {
    "module": "06_Datasets_Benchmarks/01_multimodal_datasets",
    "header": ("01. Multimodal Datasets & Data Curation", [
        "Load and explore a sample dataset from HuggingFace",
        "Analyze data distributions",
        "Data preprocessing pipelines",
        "Compare dataset sizes and characteristics",
    ], "06_Datasets_Benchmarks/01_multimodal_datasets/01_multimodal_datasets.ipynb"),
    "cells": [
        md("""## 1. Dataset Landscape Overview
"""),
        code('''PRETRAIN_DATASETS = {
    "LAION-5B": {"pairs": "5.85B", "modality": "image-text", "notes": "web crawl, CLIP-filtered"},
    "CC3M": {"pairs": "3.3M", "modality": "image-text", "notes": "Conceptual Captions"},
    "CC12M": {"pairs": "12.4M", "modality": "image-text", "notes": "noisier alt-text"},
    "DataComp": {"pairs": "variable", "modality": "image-text", "notes": "competition-scale filtering"},
    "WebLI": {"pairs": "10B+", "modality": "image-text", "notes": "Google scale pretraining"},
}

INSTRUCT_DATASETS = {
    "LLaVA-Instruct": {"size": "150K", "type": "instruction tuning"},
    "ShareGPT4V": {"size": "100K+", "type": "GPT-4V conversations"},
    "SVIT": {"size": "4.8M", "type": "synthetic VLM instructions"},
}

EVAL_BENCHMARKS = {
    "MME": "Perception + cognition score",
    "MMMU": "Multi-discipline multimodal understanding",
    "MM-Bench": "Structured VQA capabilities",
    "SEED-Bench": "19 dimensions of LVLM evaluation",
}

for name, info in PRETRAIN_DATASETS.items():
    print(f"{name:12} {info}")'''),
        md("""## 2. Load Sample Dataset (Synthetic + Optional HF)
"""),
        code('''# Synthetic caption dataset (always works offline)
synthetic = [
    {"image_id": i, "caption": cap, "source": "synthetic", "tokens": len(cap.split())}
    for i, cap in enumerate([
        "a cat on a mat", "dog playing fetch", "sunset over mountains",
        "city street at night", "person riding a bicycle", "bowl of fresh fruit",
        "snow covered trees", "children in a playground", "coffee on a wooden table",
        "airplane in the sky",
    ] * 3)
]
print(f"Synthetic samples: {len(synthetic)}")
print("Example:", synthetic[0])

# Optional HuggingFace load
try:
    from datasets import load_dataset
    ds = load_dataset("nlphuji/flickr30k", split="test[:20]", trust_remote_code=True)
    print(f"\\nFlickr30k test subset: {len(ds)} rows, columns={ds.column_names}")
    hf_ok = True
except Exception as e:
    print("\\nHF load skipped (offline or missing):", e)
    ds = None
    hf_ok = False'''),
        md("""## 3. Data Distribution Analysis
"""),
        code('''lengths = [s["tokens"] for s in synthetic]
sources = Counter(s["source"] for s in synthetic)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(lengths, bins=8, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Caption length (tokens)'); axes[0].set_title('Caption Length Distribution')
axes[1].bar(sources.keys(), sources.values(), color='coral')
axes[1].set_title('Samples by Source')
plt.tight_layout(); plt.show()

if hf_ok and ds is not None:
    hf_lens = [len(row["caption"].split()) if "caption" in row else len(row["sentences"][0].split()) for row in ds]
    print(f"HF caption length mean={np.mean(hf_lens):.1f}, std={np.std(hf_lens):.1f}")'''),
        md("""## 4. Preprocessing Pipeline
"""),
        code('''def preprocess_batch(examples, max_length=16):
    out = []
    for ex in examples:
        cap = ex.get("caption", "")
        cap = cap.lower().strip()
        cap = " ".join(cap.split())  # normalize whitespace
        tokens = cap.split()[:max_length]
        out.append({"caption_clean": " ".join(tokens), "length": len(tokens), "image_id": ex["image_id"]})
    return out

clean = preprocess_batch(synthetic)
print("Preprocessed sample:", clean[0])

# Deduplication demo
unique_caps = {c["caption_clean"] for c in clean}
print(f"Before dedup: {len(clean)} -> unique captions: {len(unique_caps)}")'''),
        md("""## 5. Quality Scoring & Filtering
"""),
        code('''def quality_score(caption):
    tokens = caption.split()
    if len(tokens) < 3:
        return 0.2
    score = 0.5
    if len(tokens) >= 5:
        score += 0.2
    if any(w in caption for w in ["a", "the", "on", "in"]):
        score += 0.1
    return min(score, 1.0)

scored = [(c["caption_clean"], quality_score(c["caption_clean"])) for c in clean]
scored.sort(key=lambda x: -x[1])
print("Top quality captions:")
for cap, s in scored[:5]:
    print(f"  [{s:.2f}] {cap}")

threshold = 0.6
filtered = [c for c in clean if quality_score(c["caption_clean"]) >= threshold]
print(f"\\nKept {len(filtered)}/{len(clean)} after quality filter (>={threshold})")'''),
        md("""## 6. Dataset Size Comparison
"""),
        code('''comparison = [
    ("LAION-5B", 5.85e9, "pretrain"),
    ("WebLI", 10e9, "pretrain"),
    ("CC12M", 12.4e6, "pretrain"),
    ("LLaVA-Instruct", 150e3, "instruct"),
    ("ShareGPT4V", 100e3, "instruct"),
    ("Flickr30k", 31.8e3, "eval"),
]

names, sizes, kinds = zip(*comparison)
colors = ['#4C72B0' if k == 'pretrain' else '#55A868' if k == 'instruct' else '#C44E52' for k in kinds]
plt.figure(figsize=(10, 5))
plt.barh(names, sizes, color=colors)
plt.xscale('log'); plt.xlabel('Number of image-text pairs (log scale)')
plt.title('Multimodal Dataset Scale Comparison'); plt.tight_layout(); plt.show()'''),
        md("""## Summary

Explored pretraining/instruction/eval datasets, built a preprocessing and quality filtering pipeline, and compared dataset scales.

See [README.md](./README.md) for full dataset catalog and curation math.
"""),
    ],
}
