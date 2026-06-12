# Utils — Shared Visualization & Training Utilities

> **API Reference** for the helper modules used across all 17 notebooks.

---

## Overview

```
  utils/
  ├── __init__.py           # Makes utils a Python package
  ├── visualization.py      # 10+ plotting & diagram functions
  └── helpers.py            # Training loop, device utils, data generators
```

Every notebook imports these utilities at the top:

```python
import sys
sys.path.append('..')
from utils.visualization import *
from utils.helpers import *
```

---

## `visualization.py` — Plotting & Diagram Functions

All 54 visualizations in this course are generated **programmatically**. Modify the code and re-run to customize any diagram.

### Style Setup

```python
set_style()
```

Configures a consistent matplotlib theme across all notebooks:

| Setting | Value |
|---------|-------|
| Font sizes | Title: 14, Labels: 12, Ticks: 10 |
| Grid | Light gray, dashed |
| DPI | 150 (crisp on all screens) |
| Figure size | 10 × 6 (default) |

---

### Architecture Diagrams

| Function | Signature | What It Draws |
|----------|-----------|---------------|
| `draw_architecture_block` | `(ax, x, y, w, h, label, color)` | Colored rounded box with label |
| `draw_arrow` | `(ax, start, end)` | Arrow connecting two points |
| `plot_multimodal_overview` | `()` | Full multimodal pipeline (Img/Txt/Audio → Fusion → Out) |
| `draw_fusion_comparison` | `()` | Side-by-side Early/Late/Cross fusion |
| `draw_lora_diagram` | `()` | LoRA architecture ($W_0 x + BAx$) |

---

### Data Visualization

#### `plot_similarity_matrix(emb_a, emb_b, labels)`

Computes and renders the cosine similarity matrix:

$$S_{ij} = \frac{\mathbf{a}_i^\top \mathbf{b}_j}{\|\mathbf{a}_i\| \|\mathbf{b}_j\|}$$

Used in CLIP notebooks (02.01, 04.04) to show how well images and texts align.

#### `plot_attention_heatmap(weights, x_labels, y_labels)`

Renders attention weights $A \in \mathbb{R}^{T_Q \times T_K}$ as an annotated heatmap:

$$A_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_l \exp(q_i^\top k_l / \sqrt{d_k})}$$

Each row sums to 1.0 (softmax normalization). Used in VQA (02.03) to show which image patches text tokens attend to.

#### `plot_training_curves(train_loss, val_loss, ...)`

Plots training and validation loss curves on the same axes. Optionally overlays learning rate schedule on a secondary y-axis.

#### `plot_embedding_space(embeddings, labels, modalities)`

Applies t-SNE dimensionality reduction:

$$\text{t-SNE}: \mathbb{R}^D \to \mathbb{R}^2$$

and renders a scatter plot with color-coded modalities (image, text, audio).

#### `plot_parameter_comparison(full, trainable, names)`

Horizontal bar chart comparing total vs trainable parameters — makes the LoRA savings dramatic:

```
  Full Layer     |==============================================| 589,824
  LoRA (r=8)     |=|                                              12,288

                                                                  97.9% fewer!
```

---

## `helpers.py` — Training & Data Utilities

### `get_device() → torch.device`

```
  Priority:  CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU (fallback)
```

### `count_parameters(model) → dict`

Returns total, trainable, and frozen parameter counts:

$$\text{trainable \%} = \frac{|\theta_{\text{train}}|}{|\theta_{\text{total}}|} \times 100$$

### `estimate_memory(model, input_shape) → dict`

Estimates GPU memory using:

$$\text{Memory} \approx \underbrace{4|\theta|}_{\text{weights}} + \underbrace{4|\theta|}_{\text{gradients}} + \underbrace{8|\theta|}_{\text{Adam states}} + \underbrace{\text{activations}}_{\text{data-dependent}}$$

(in bytes, for fp32 training with Adam optimizer)

### `SimpleTrainer`

```python
trainer = SimpleTrainer(model, optimizer, device)
history = trainer.fit(train_loader, val_loader, epochs=10, loss_fn=loss_fn)
```

Features:

| Feature | Description |
|---------|-------------|
| Progress bar | tqdm-based, shows batch progress |
| Validation | Automatic eval after each epoch |
| Gradient clipping | Configurable `max_norm` |
| LR scheduling | Supports any PyTorch scheduler |
| History logging | Returns dict with loss, accuracy, time per epoch |

### `create_synthetic_image_text_pairs(n_samples, img_size, n_classes)`

Generates synthetic training data with **no downloads needed**:

| Class | Image | Text |
|-------|-------|------|
| 0 | Red circle on gray | "a red circle" |
| 1 | Blue square on gray | "a blue square" |
| 2 | Green triangle on gray | "a green triangle" |
| ... | ... | ... |

Default: 5 classes × 40 samples = 200 pairs, each image $32 \times 32 \times 3$.

### `ImageTextDataset`

Wraps images and texts into a PyTorch `Dataset`:

```python
dataset = ImageTextDataset(images, texts, transform=..., tokenizer=...)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Quick Reference

| What You Need | Function | File |
|---------------|----------|------|
| Set plot style | `set_style()` | visualization.py |
| Draw architecture | `draw_architecture_block()` | visualization.py |
| Plot heatmap | `plot_attention_heatmap()` | visualization.py |
| Plot similarity | `plot_similarity_matrix()` | visualization.py |
| Plot training | `plot_training_curves()` | visualization.py |
| Detect device | `get_device()` | helpers.py |
| Count params | `count_parameters()` | helpers.py |
| Train model | `SimpleTrainer.fit()` | helpers.py |
| Synthetic data | `create_synthetic_image_text_pairs()` | helpers.py |
| Dataset wrapper | `ImageTextDataset` | helpers.py |
