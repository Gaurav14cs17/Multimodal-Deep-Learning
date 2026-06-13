"""
Visualization utilities for multimodal learning notebooks.
All diagrams are generated programmatically - no external images needed.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
from typing import List, Optional, Dict, Tuple
import torch


def set_style():
    """Set consistent visual style for all notebooks."""
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
    })
    sns.set_palette("husl")


def draw_architecture_block(ax, x, y, w, h, label, color='#4ECDC4', fontsize=10):
    """Draw a rounded rectangle block for architecture diagrams."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='#2C3E50',
        linewidth=1.5, alpha=0.85
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')
    return box


def draw_arrow(ax, start, end, color='#2C3E50', style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def plot_multimodal_overview():
    """Draw the multimodal architecture overview diagram."""
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Multimodal Architecture Overview', fontsize=18, fontweight='bold', pad=20)

    colors = {
        'image': '#E74C3C', 'text': '#3498DB', 'audio': '#2ECC71',
        'fusion': '#9B59B6', 'output': '#F39C12', 'encoder': '#1ABC9C'
    }

    draw_architecture_block(ax, 2, 6.5, 2.5, 1, 'Image\n(pixels)', colors['image'])
    draw_architecture_block(ax, 7, 6.5, 2.5, 1, 'Text\n(tokens)', colors['text'])
    draw_architecture_block(ax, 12, 6.5, 2.5, 1, 'Audio\n(spectrogram)', colors['audio'])

    draw_architecture_block(ax, 2, 4.5, 2.5, 1, 'Vision Encoder\n(ViT / CNN)', colors['encoder'])
    draw_architecture_block(ax, 7, 4.5, 2.5, 1, 'Text Encoder\n(BERT / GPT)', colors['encoder'])
    draw_architecture_block(ax, 12, 4.5, 2.5, 1, 'Audio Encoder\n(Whisper)', colors['encoder'])

    for x in [2, 7, 12]:
        draw_arrow(ax, (x, 6.0), (x, 5.1))

    draw_architecture_block(ax, 7, 2.5, 8, 1.2, 'Fusion Layer\n(Cross-Attention / Concatenation / Gating)', colors['fusion'], fontsize=12)

    for x in [2, 7, 12]:
        draw_arrow(ax, (x, 4.0), (x if x == 7 else (4.5 if x == 2 else 9.5), 3.2))

    draw_architecture_block(ax, 7, 0.8, 4, 0.9, 'Task Output (Classification / Generation)', colors['output'], fontsize=10)
    draw_arrow(ax, (7, 1.9), (7, 1.35))

    plt.tight_layout()
    return fig


def plot_attention_heatmap(attention_weights, x_labels=None, y_labels=None,
                           title='Attention Weights'):
    """Visualize attention weights as a heatmap."""
    set_style()
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_weights, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=x_labels, yticklabels=y_labels,
                ax=ax, cbar_kws={'label': 'Attention Weight'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Keys')
    ax.set_ylabel('Queries')
    plt.tight_layout()
    return fig


def plot_training_curves(train_losses, val_losses=None, train_accs=None,
                          val_accs=None, title='Training Progress'):
    """Plot training loss and accuracy curves."""
    set_style()
    n_plots = 1 + (1 if train_accs is not None else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(train_losses, label='Train Loss', color='#E74C3C', linewidth=2)
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss', color='#3498DB', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()

    if train_accs is not None:
        axes[1].plot(train_accs, label='Train Acc', color='#2ECC71', linewidth=2)
        if val_accs:
            axes[1].plot(val_accs, label='Val Acc', color='#F39C12', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curve')
        axes[1].legend()

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_similarity_matrix(embeddings_a, embeddings_b, labels_a=None,
                            labels_b=None, title='Cosine Similarity Matrix'):
    """Plot cosine similarity between two sets of embeddings."""
    set_style()
    if isinstance(embeddings_a, torch.Tensor):
        embeddings_a = embeddings_a.detach().cpu().numpy()
    if isinstance(embeddings_b, torch.Tensor):
        embeddings_b = embeddings_b.detach().cpu().numpy()

    norm_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
    norm_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
    sim_matrix = norm_a @ norm_b.T

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=-1, vmax=1, center=0,
                xticklabels=labels_b, yticklabels=labels_a, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Text Embeddings')
    ax.set_ylabel('Image Embeddings')
    plt.tight_layout()
    return fig


def plot_embedding_space(embeddings, labels, modalities=None,
                          title='Embedding Space (t-SNE)'):
    """Visualize embeddings in 2D using t-SNE."""
    set_style()
    from sklearn.manifold import TSNE

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    if modalities is not None:
        unique_modalities = list(set(modalities))
        markers = ['o', 's', '^', 'D', 'v']
        for i, mod in enumerate(unique_modalities):
            mask = [m == mod for m in modalities]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      label=mod, marker=markers[i % len(markers)], s=100, alpha=0.7)
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=range(len(labels)),
                           cmap='tab10', s=100, alpha=0.7)

    for i, label in enumerate(labels):
        ax.annotate(label, (coords[i, 0] + 0.5, coords[i, 1] + 0.5), fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_parameter_comparison(full_params, trainable_params, method_names):
    """Bar chart comparing full vs trainable parameters for different methods."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(method_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, [p/1e6 for p in full_params],
                   width, label='Total Params', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, [p/1e6 for p in trainable_params],
                   width, label='Trainable Params', color='#E74C3C', alpha=0.8)

    for bar, full, train in zip(bars2, full_params, trainable_params):
        pct = train / full * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#E74C3C')

    ax.set_xlabel('Method')
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Parameter Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    plt.tight_layout()
    return fig


def draw_lora_diagram():
    """Draw LoRA architecture diagram."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('LoRA: Low-Rank Adaptation', fontsize=18, fontweight='bold', pad=20)

    draw_architecture_block(ax, 5, 7, 3, 0.8, 'Input x', '#34495E')

    draw_architecture_block(ax, 3, 5, 2.5, 0.8, 'W (frozen)\nd × d', '#3498DB')
    ax.text(3, 5.7, 'FROZEN', fontsize=8, ha='center', color='#3498DB', style='italic')

    draw_architecture_block(ax, 7, 5.5, 1.8, 0.6, 'A: d × r', '#E74C3C')
    draw_architecture_block(ax, 7, 4.5, 1.8, 0.6, 'B: r × d', '#E74C3C')
    ax.text(8.5, 5.0, f'rank r << d\n(r=4,8,16)', fontsize=9, ha='center',
            color='#E74C3C', style='italic')

    draw_arrow(ax, (5, 6.6), (3, 5.5))
    draw_arrow(ax, (5, 6.6), (7, 5.9))
    draw_arrow(ax, (7, 5.2), (7, 4.9))

    draw_architecture_block(ax, 5, 3, 3, 0.8, 'h = Wx + BAx', '#2ECC71')
    draw_arrow(ax, (3, 4.6), (4.2, 3.5))
    draw_arrow(ax, (7, 4.2), (5.8, 3.5))

    draw_architecture_block(ax, 5, 1.5, 3, 0.8, 'Output h', '#34495E')
    draw_arrow(ax, (5, 2.6), (5, 2.0))

    ax.text(1, 1, 'Only A and B are trained!\nW stays frozen.',
            fontsize=11, color='#E74C3C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FADBD8', alpha=0.8))

    plt.tight_layout()
    return fig


def draw_fusion_comparison():
    """Draw comparison of fusion strategies."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax in axes:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.axis('off')

    ax = axes[0]
    ax.set_title('Early Fusion', fontsize=14, fontweight='bold')
    draw_architecture_block(ax, 1.5, 7, 2, 0.7, 'Image', '#E74C3C')
    draw_architecture_block(ax, 4.5, 7, 2, 0.7, 'Text', '#3498DB')
    draw_architecture_block(ax, 3, 5.5, 4, 0.7, 'Concatenate', '#F39C12')
    draw_architecture_block(ax, 3, 4, 4, 0.7, 'Joint Encoder', '#1ABC9C')
    draw_architecture_block(ax, 3, 2.5, 3, 0.7, 'Output', '#9B59B6')
    draw_arrow(ax, (1.5, 6.6), (2, 5.9))
    draw_arrow(ax, (4.5, 6.6), (4, 5.9))
    draw_arrow(ax, (3, 5.1), (3, 4.4))
    draw_arrow(ax, (3, 3.6), (3, 2.9))

    ax = axes[1]
    ax.set_title('Late Fusion', fontsize=14, fontweight='bold')
    draw_architecture_block(ax, 1.5, 7, 2, 0.7, 'Image', '#E74C3C')
    draw_architecture_block(ax, 4.5, 7, 2, 0.7, 'Text', '#3498DB')
    draw_architecture_block(ax, 1.5, 5.5, 2, 0.7, 'Img Encoder', '#1ABC9C')
    draw_architecture_block(ax, 4.5, 5.5, 2, 0.7, 'Txt Encoder', '#1ABC9C')
    draw_architecture_block(ax, 3, 4, 4, 0.7, 'Fuse (concat/add)', '#F39C12')
    draw_architecture_block(ax, 3, 2.5, 3, 0.7, 'Output', '#9B59B6')
    draw_arrow(ax, (1.5, 6.6), (1.5, 5.9))
    draw_arrow(ax, (4.5, 6.6), (4.5, 5.9))
    draw_arrow(ax, (1.5, 5.1), (2.2, 4.4))
    draw_arrow(ax, (4.5, 5.1), (3.8, 4.4))
    draw_arrow(ax, (3, 3.6), (3, 2.9))

    ax = axes[2]
    ax.set_title('Cross-Modal Fusion', fontsize=14, fontweight='bold')
    draw_architecture_block(ax, 1.5, 7, 2, 0.7, 'Image', '#E74C3C')
    draw_architecture_block(ax, 4.5, 7, 2, 0.7, 'Text', '#3498DB')
    draw_architecture_block(ax, 1.5, 5.5, 2, 0.7, 'Img Encoder', '#1ABC9C')
    draw_architecture_block(ax, 4.5, 5.5, 2, 0.7, 'Txt Encoder', '#1ABC9C')
    draw_architecture_block(ax, 3, 4, 4, 0.7, 'Cross-Attention', '#E74C3C')
    draw_architecture_block(ax, 3, 2.5, 3, 0.7, 'Output', '#9B59B6')
    draw_arrow(ax, (1.5, 6.6), (1.5, 5.9))
    draw_arrow(ax, (4.5, 6.6), (4.5, 5.9))
    draw_arrow(ax, (1.5, 5.1), (2.2, 4.4))
    draw_arrow(ax, (4.5, 5.1), (3.8, 4.4))
    draw_arrow(ax, (3, 3.6), (3, 2.9))
    ax.annotate('', xy=(4, 4.7), xytext=(2, 4.7),
                arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2, ls='--'))

    plt.tight_layout()
    return fig
