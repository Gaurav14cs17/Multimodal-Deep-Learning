"""
Helper utilities for data loading, training, and model management.
Optimized for low-compute environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple, Callable
from tqdm import tqdm
import numpy as np
import time


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU (notebooks are optimized for this!)")
    return device


def count_parameters(model, print_table=True):
    """Count and display model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    if print_table:
        print(f"{'='*50}")
        print(f"{'Parameter Summary':^50}")
        print(f"{'='*50}")
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        print(f"  Frozen parameters:    {frozen:>12,}")
        print(f"  Trainable %:          {trainable/total*100:>11.2f}%")
        print(f"{'='*50}")

    return {'total': total, 'trainable': trainable, 'frozen': frozen}


def estimate_memory(model, input_shape, batch_size=1, dtype=torch.float32):
    """Estimate memory usage for a model."""
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_mem = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

    input_mem = np.prod(input_shape) * batch_size * 4  # float32
    # Rough activation memory estimate
    act_mem = param_mem * 2

    total = param_mem + grad_mem + input_mem + act_mem

    print(f"Memory Estimate (batch_size={batch_size}):")
    print(f"  Parameters:   {param_mem / 1e6:.1f} MB")
    print(f"  Gradients:    {grad_mem / 1e6:.1f} MB")
    print(f"  Input:        {input_mem / 1e6:.1f} MB")
    print(f"  Activations:  ~{act_mem / 1e6:.1f} MB")
    print(f"  Total:        ~{total / 1e6:.1f} MB")
    return total


class SimpleTrainer:
    """Minimal training loop with logging and visualization support."""

    def __init__(self, model, optimizer, device='cpu', scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': [], 'epoch_time': []
        }

    def train_epoch(self, dataloader, loss_fn, compute_acc=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc='Training', leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()} if isinstance(batch, dict) else batch

            self.optimizer.zero_grad()

            if isinstance(batch, dict):
                loss = loss_fn(self.model, batch)
            else:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                if compute_acc:
                    total_correct += compute_acc(outputs, targets)
                total_samples += targets.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples if total_samples > 0 else None
        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self, dataloader, loss_fn, compute_acc=None):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                loss = loss_fn(self.model, batch)
            else:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                if compute_acc:
                    total_correct += compute_acc(outputs, targets)
                total_samples += targets.size(0)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples if total_samples > 0 else None
        return avg_loss, avg_acc

    def fit(self, train_loader, val_loader=None, epochs=10, loss_fn=None,
            compute_acc=None, verbose=True):
        """Full training loop."""
        for epoch in range(epochs):
            start = time.time()

            train_loss, train_acc = self.train_epoch(train_loader, loss_fn, compute_acc)
            self.history['train_loss'].append(train_loss)
            if train_acc is not None:
                self.history['train_acc'].append(train_acc)

            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, loss_fn, compute_acc)
                self.history['val_loss'].append(val_loss)
                if val_acc is not None:
                    self.history['val_acc'].append(val_acc)

            if self.scheduler:
                self.scheduler.step()
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            elapsed = time.time() - start
            self.history['epoch_time'].append(elapsed)

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}"
                if train_acc is not None:
                    msg += f" | Train Acc: {train_acc:.4f}"
                if val_loader:
                    msg += f" | Val Loss: {val_loss:.4f}"
                    if val_acc is not None:
                        msg += f" | Val Acc: {val_acc:.4f}"
                msg += f" | Time: {elapsed:.1f}s"
                print(msg)

        return self.history


class ImageTextDataset(Dataset):
    """Simple dataset for image-text pairs."""

    def __init__(self, images, texts, transform=None, tokenizer=None, max_length=77):
        self.images = images
        self.texts = texts
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]

        if self.transform:
            image = self.transform(image)

        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_length,
                                   padding='max_length', truncation=True,
                                   return_tensors='pt')
            return {
                'image': image,
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'text': text
            }
        return {'image': image, 'text': text}


def create_synthetic_image_text_pairs(n_samples=100, img_size=32, n_classes=5):
    """Create synthetic image-text pairs for testing (no downloads needed)."""
    images = []
    texts = []
    labels = []

    class_configs = [
        ('red circle', [1.0, 0.2, 0.2]),
        ('blue square', [0.2, 0.2, 1.0]),
        ('green triangle', [0.2, 1.0, 0.2]),
        ('yellow star', [1.0, 1.0, 0.2]),
        ('purple diamond', [0.8, 0.2, 0.8]),
    ]

    for i in range(n_samples):
        cls = i % min(n_classes, len(class_configs))
        name, color = class_configs[cls]

        img = torch.zeros(3, img_size, img_size)
        cx, cy = img_size // 2, img_size // 2
        r = img_size // 4

        y_coords, x_coords = torch.meshgrid(
            torch.arange(img_size), torch.arange(img_size), indexing='ij'
        )
        mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) < r ** 2

        for c in range(3):
            img[c][mask] = color[c]
        img += torch.randn_like(img) * 0.1
        img = img.clamp(0, 1)

        templates = [
            f"a photo of a {name}",
            f"this is a {name}",
            f"an image showing a {name}",
            f"a {name} on a dark background",
        ]
        text = templates[i % len(templates)]

        images.append(img)
        texts.append(text)
        labels.append(cls)

    return images, texts, labels
