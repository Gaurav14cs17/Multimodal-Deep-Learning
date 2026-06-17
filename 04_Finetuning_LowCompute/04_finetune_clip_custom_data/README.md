# 04 — Finetune CLIP on Your Own Custom Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/04_finetune_clip_custom_data/04_finetune_clip_custom_data.ipynb)

> **Time:** ~45 minutes | **Difficulty:** Intermediate | **GPU Required:** Recommended

---

## What You'll Learn

This is the **practical capstone** of Module 04 — take everything you've learned and apply it to YOUR data.

- Prepare your own image-text dataset
- Finetune OpenCLIP with LoRA (low compute)
- Evaluate with retrieval metrics (Recall@1, Recall@5, Recall@10)
- Compare: frozen CLIP vs full FT vs LoRA FT
- Export and use your finetuned model

---

## End-to-End Pipeline

```
  ┌───────────────────────────────────────────────────────────────┐
  │                    CLIP Finetuning Pipeline                    │
  │                                                                │
  │  Step 1: Data Preparation                                     │
  │  ┌──────────────────┐                                         │
  │  │ images/           │     ┌──────────────────┐               │
  │  │   cat_01.jpg      │ ──► │ Custom Dataset   │               │
  │  │   dog_02.jpg      │     │  __getitem__():  │               │
  │  │ captions.csv      │     │  (image, text)   │               │
  │  │   "a tabby cat"   │     └────────┬─────────┘               │
  │  └──────────────────┘              │                          │
  │                                     ▼                          │
  │  Step 2: Model Setup                                          │
  │  ┌──────────────────────────────────────────┐                 │
  │  │  OpenCLIP ViT-B/32 (pretrained)          │                 │
  │  │  ├── Image Encoder (frozen + LoRA)       │                 │
  │  │  ├── Text Encoder (frozen + LoRA)        │                 │
  │  │  └── Projection Heads (trainable)        │                 │
  │  └──────────────────┬───────────────────────┘                 │
  │                     │                                          │
  │                     ▼                                          │
  │  Step 3: Training                                             │
  │  ┌──────────────────────────────────────────┐                 │
  │  │  InfoNCE Loss + AdamW + Cosine LR        │                 │
  │  │  Gradient Accumulation (effective BS=256) │                 │
  │  │  Mixed Precision (fp16)                   │                 │
  │  │  ~30 minutes on 1 GPU                    │                 │
  │  └──────────────────┬───────────────────────┘                 │
  │                     │                                          │
  │                     ▼                                          │
  │  Step 4: Evaluation + Export                                  │
  │  ┌──────────────────────────────────────────┐                 │
  │  │  Recall@1/5/10, Median Rank              │                 │
  │  │  LoRA merge → single checkpoint          │                 │
  │  │  Deploy for your domain                  │                 │
  │  └──────────────────────────────────────────┘                 │
  └───────────────────────────────────────────────────────────────┘
```

---

## Key Equations

### 1. Finetuning Objective — Same as Pretraining

$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} + \log\frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}\right]
$$

but now trained on **your domain-specific data** instead of web-scale data.

### 2. Evaluation Metric — Recall@K

Given a query (image or text), retrieve the top $K$ candidates from the other modality and check if the correct match is among them:

$$
\text{Recall@}K = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[\text{rank}(i) \leq K\right]
$$

where $\text{rank}(i)$ is the rank of the correct match for query $i$ among all candidates sorted by similarity.

**Median Rank (MedR):**

$$
\text{MedR} = \text{median}\!\left(\{\text{rank}(1), \text{rank}(2), \ldots, \text{rank}(N)\}\right)
$$

Lower is better. MedR = 1 means the correct answer is almost always ranked first.

### 3. Learning Rate Schedule — Cosine with Warmup

$$
\eta(t) =
\begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warm}}} & \text{if } t < T_{\text{warm}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{t - T_{\text{warm}}}{T - T_{\text{warm}}} \pi\right)\right) & \text{otherwise}
\end{cases}
$$

```
  Learning rate schedule:

  η_max ┤    ╱─────────╲
        │   ╱             ╲
        │  ╱               ╲
        │ ╱                  ╲
        │╱                     ╲
  η_min ┤                       ╲_____
        └──┴──────────┴───────────┴──
         0  T_warm                  T
            Warmup     Cosine decay
```

### 4. Gradient Accumulation — Simulating Large Batches

If your GPU can only fit $B_{\text{micro}}$ samples but you want effective batch size $B_{\text{eff}}$:

$$
\text{Accumulation steps} = \frac{B_{\text{eff}}}{B_{\text{micro}}}
$$

$$
g_{\text{accumulated}} = \frac{1}{K}\sum_{k=1}^{K} g_k
$$

where $K$ is the number of accumulation steps and $g_k$ is the gradient from micro-batch $k$.

---

## Recommended Hyperparameters

| Parameter | Frozen CLIP | LoRA FT | Full FT |
|-----------|------------|---------|---------|
| Learning rate | — | $2 \times 10^{-4}$ | $5 \times 10^{-6}$ |
| LoRA rank $r$ | — | 8 | — |
| LoRA $\alpha$ | — | 16 | — |
| Batch size | — | 256 (via accum.) | 256 |
| Epochs | — | 10–30 | 5–10 |
| Warmup | — | 10% of steps | 5% |
| Weight decay | — | 0.01 | 0.01 |
| Temperature $\tau$ | 0.07 (fixed) | learnable | learnable |

---

## Expected Results

| Method | Recall@1 | Recall@5 | Recall@10 | MedR | Training Time |
|--------|----------|----------|-----------|------|--------------|
| Frozen CLIP | ~45% | ~72% | ~82% | 2.0 | 0 min |
| LoRA FT ($r=8$) | ~78% | ~94% | ~97% | 1.0 | ~30 min |
| Full FT | ~82% | ~96% | ~98% | 1.0 | ~4 hours |

**Key insight:** LoRA achieves ~95% of full FT quality in ~12% of the training time.

---

## Dataset Format

Your custom data should follow this structure:

```
  my_dataset/
  ├── images/
  │   ├── 0001.jpg
  │   ├── 0002.jpg
  │   └── ...
  └── captions.csv
      ┌──────────────────────────────────────┐
      │ image_path,caption                    │
      │ images/0001.jpg,"a red sports car"    │
      │ images/0002.jpg,"sunset over ocean"   │
      │ ...                                   │
      └──────────────────────────────────────┘
```

### Data Quality Tips

| Tip | Why |
|-----|-----|
| Min 500 pairs, ideally 2K+ | LoRA needs less data but not trivially small |
| Diverse negatives | Avoid all images being same category |
| Specific captions | "A red 2020 Tesla Model 3" > "a car" |
| Clean images | Remove duplicates, blurry photos |
| Balanced classes | Avoid 90% one category |

---

## What You'll Build

- Custom dataset loader for image-text pairs
- LoRA-wrapped OpenCLIP model
- Training loop with cosine LR and gradient accumulation
- Evaluation: retrieval accuracy before and after finetuning
- LoRA merging and model export for deployment

---

## Prerequisites

- Module 04 Notebooks 01–03 (LoRA, QLoRA, Adapters)
- Module 02 Notebook 01 (CLIP architecture)
- Own image-text dataset (or use the synthetic one provided)

---

## 🔬 Worked Examples in the Notebook

### Domain Adaptation — Before vs After LoRA Finetuning
- Simulate Recall@K on domain-specific data (medical, product, etc.)
- Compare: generic CLIP (R@1 ~25%) vs LoRA-finetuned (R@1 ~85%)
- Similarity distribution: matching pairs shift from ~0.3 to ~0.9
- Real-world pipeline: collect → clean → format → augment → train → evaluate → deploy

> 💡 **Run the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/Multimodal-Deep-Learning/blob/main/04_Finetuning_LowCompute/04_finetune_clip_custom_data/04_finetune_clip_custom_data.ipynb)

---

## 📄 Paper Figures in the Notebook

| Figure | Paper | Year | Key Concept |
|--------|-------|------|-------------|
| CLIP Domain Adaptation Pipeline | CLIP + LoRA + OpenCLIP | 2021-22 | General → domain-specific via LoRA adapters |

### Domain Adaptation Applications

- **Medical imaging:** BiomedCLIP, PubMedCLIP
- **Remote sensing:** RemoteCLIP for satellite imagery
- **Retail:** Product search and visual grounding
- **Industrial:** Defect detection, quality control

---

## Next Step

**[01_llava_architecture](../../05_Advanced_Topics/01_llava_architecture/)** — How LLaVA connects vision to LLMs
