#!/usr/bin/env python3
"""
End‑to‑end fruit sorting with ResNet teacher and ViT student using knowledge distillation.

This script implements a complete pipeline for training a convolutional teacher model
(ResNet50) on a fruit classification dataset with configurable label noise and
Albumentations-based augmentations, then transferring its knowledge to a vision
transformer (ViT) student via knowledge distillation.  Both models support
instance‑dependent, symmetric, class‑conditional and mixed noise injection.  The
student can be trained on the same dataset as the teacher or on a different
dataset to assess transferability.  The pipeline includes warm‑up and refinement
stages for the teacher, and an optional cross‑entropy plus KL‑divergence loss
for the student.

Knowledge distillation between CNNs and ViTs is beneficial when training data
are limited; recent research has demonstrated that ViTs trained as students of
CNN teachers achieve higher accuracy than standalone ViTs【835975376916920†L186-L199】.
This script follows that approach by computing a weighted sum of the
student's cross‑entropy loss and the KL divergence between the student and
teacher logits (with temperature scaling), enabling the student to match the
teacher's predictions while learning from ground truth labels.

Example usage:

```bash
python fruit_sorting_transfer_distill.py \
  --teacher_data_root /data/Fruits-360/Training \
  --student_data_root /data/FruitRecognition \
  --teacher_noise_mode symmetric --teacher_noise_rate 0.2 \
  --student_noise_mode symmetric --student_noise_rate 0.1 \
  --teacher_warmup_epochs 6 --teacher_refine_epochs 6 \
  --student_epochs 20 \
  --distill_alpha 0.7 --distill_temperature 3.0 \
  --output_dir runs/resnet_vit_distill
```

Dependencies:
```
pip install torch torchvision albumentations opencv-python scikit-learn
```
"""

import os
import cv2
import json
import glob
import random
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib

plt = None


def setup_matplotlib(show_plots: bool = False) -> None:
    """Configure matplotlib backend lazily so we can optionally show plots."""
    global plt
    if plt is not None:
        return
    backend_env = os.environ.get('MPLBACKEND')
    desired = backend_env or ('TkAgg' if show_plots else 'Agg')
    try:
        matplotlib.use(desired)
    except Exception as exc:
        if desired != 'Agg':
            print(f"[Plot] Could not use backend '{desired}': {exc}. Falling back to Agg.")
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt_mod  # pylint: disable=import-outside-toplevel
    plt = plt_mod


# -----------------------------------------------------------------------------
# Noise injection utilities
# -----------------------------------------------------------------------------
def inject_symmetric_noise(y: np.ndarray, rate: float, rng: np.random.Generator, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Flip labels uniformly to any other class at the desired rate."""
    y_noisy = y.copy()
    n = len(y_noisy)
    if rate <= 0 or num_classes < 2:
        return y_noisy, np.array([], dtype=np.int32)
    flip_count = int(rate * n)
    if flip_count == 0:
        return y_noisy, np.array([], dtype=np.int32)
    idx = rng.choice(n, size=flip_count, replace=False)
    random_labels = rng.integers(0, num_classes - 1, size=flip_count, dtype=np.int32)
    random_labels += (random_labels >= y_noisy[idx]).astype(np.int32)
    y_noisy[idx] = random_labels
    return y_noisy, idx


def inject_ccn_noise(y: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Class‑conditional noise using confusion matrix P where P[i,j] = Pr(y_obs=j | y_true=i).
    """
    P = np.asarray(P, dtype=np.float32)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Confusion matrix must be square")
    num_classes = P.shape[0]
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Each row of confusion matrix must sum to 1")
    y_noisy = y.copy()
    idx_changed: List[int] = []
    for i in range(len(y_noisy)):
        true = y[i]
        noisy = rng.choice(num_classes, p=P[int(true)])
        if noisy != true:
            idx_changed.append(i)
        y_noisy[i] = noisy
    return y_noisy, np.array(idx_changed, dtype=np.int32)


def inject_instance_dep_noise(y: np.ndarray, lap_vars: np.ndarray, ents: np.ndarray, max_rate: float, rng: np.random.Generator, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Instance‑dependent noise. Flip probability increases with image difficulty.
    Use normalized blur score and entropy to compute per‑sample flip prob in [0, max_rate].
    """
    y_noisy = y.copy()
    if num_classes < 2 or max_rate <= 0:
        return y_noisy, np.array([], dtype=np.int32)
    lap = np.array(lap_vars)
    ent = np.array(ents)
    lap_clipped = np.clip(lap, np.percentile(lap, 5), np.percentile(lap, 95))
    lap_norm = 1.0 - (lap_clipped - lap_clipped.min()) / (lap_clipped.max() - lap_clipped.min() + 1e-8)
    ent_clipped = np.clip(ent, np.percentile(ent, 5), np.percentile(ent, 95))
    ent_norm = (ent_clipped - ent_clipped.min()) / (ent_clipped.max() - ent_clipped.min() + 1e-8)
    diff = 0.6 * lap_norm + 0.4 * ent_norm
    flip_prob = max_rate * diff
    flips = rng.random(len(y)) < flip_prob
    idx_changed = np.where(flips)[0]
    if len(idx_changed) == 0:
        return y_noisy, idx_changed
    new_labels = rng.integers(0, num_classes - 1, size=len(idx_changed), dtype=np.int32)
    new_labels += (new_labels >= y_noisy[idx_changed]).astype(np.int32)
    y_noisy[idx_changed] = new_labels
    return y_noisy, idx_changed


def inject_mixed_noise(y: np.ndarray, lap_vars: np.ndarray, ents: np.ndarray, cfg: Dict, rng: np.random.Generator, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixed noise with weights across modes: cfg contains keys 'symmetric', 'ccn', 'idn' with parameters and weights.
    """
    weights: List[float] = []
    modes: List[str] = []
    for k in ["symmetric", "ccn", "idn"]:
        if k in cfg and cfg[k].get("weight", 0) > 0:
            weights.append(cfg[k]["weight"])
            modes.append(k)
    if not weights:
        return y.copy(), np.array([], dtype=np.int32)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    y_noisy = y.copy()
    n = len(y)
    counts = np.floor(weights * n).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        order = np.argsort(weights)[::-1]
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    perm = rng.permutation(n)
    start = 0
    changed_all: List[int] = []
    for mode, count in zip(modes, counts):
        if count == 0:
            continue
        idx = perm[start:start + count]
        start += count
        if mode == "symmetric":
            yn, ch = inject_symmetric_noise(y_noisy[idx], cfg["symmetric"].get("rate", 0.0), rng, num_classes)
        elif mode == "ccn":
            P = np.array(cfg["ccn"].get("P"), dtype=np.float32)
            yn, ch = inject_ccn_noise(y_noisy[idx], P, rng)
        elif mode == "idn":
            yn, ch = inject_instance_dep_noise(y_noisy[idx], lap_vars[idx], ents[idx], cfg["idn"].get("max_rate", 0.0), rng, num_classes)
        else:
            continue
        y_noisy[idx] = yn
        if len(ch) > 0:
            changed_all.extend(idx[ch])
    return y_noisy, np.array(changed_all, dtype=np.int32)


# -----------------------------------------------------------------------------
# Difficulty metrics
# -----------------------------------------------------------------------------
def compute_difficulty_metrics(paths: List[str], image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Laplacian variance and entropy once per path."""
    lap_vars = np.zeros(len(paths), dtype=np.float32)
    ents = np.zeros(len(paths), dtype=np.float32)
    for idx, path in enumerate(paths):
        bgr = cv2.imread(path)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        bgr_resized = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
        gray0 = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2GRAY)
        lap_vars[idx] = float(cv2.Laplacian(gray0, cv2.CV_64F).var())
        hist = cv2.calcHist([gray0], [0], None, [64], [0, 256])
        p = hist / (np.sum(hist) + 1e-8)
        ents[idx] = float(-np.sum(p * np.log(p + 1e-12)))
    return lap_vars, ents


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------
def _has_class_folders_with_files(root: str) -> bool:
    """Return True if root contains ≥1 class folder that itself has image files."""
    if not os.path.isdir(root):
        return False
    for entry in os.listdir(root):
        if entry.startswith('.'):
            continue
        class_dir = os.path.join(root, entry)
        if not os.path.isdir(class_dir):
            continue
        files = glob.glob(os.path.join(class_dir, '*'))
        if any(os.path.isfile(p) for p in files):
            return True
    return False


def resolve_class_root(root: str) -> str:
    """Automatically pick a split folder (e.g., Training) that actually holds class subfolders."""
    root = os.path.abspath(root)
    if _has_class_folders_with_files(root):
        return root
    for candidate in ['Training', 'training', 'Train', 'train']:
        cand_path = os.path.join(root, candidate)
        if _has_class_folders_with_files(cand_path):
            print(f"[Dataset] Using '{candidate}' split found under {root}")
            return cand_path
    return root


def list_images_with_labels(root: str, class_names: Optional[List[str]] = None) -> Tuple[List[str], np.ndarray, List[str]]:
    """Return (paths, labels, class_names) for a directory-structured dataset."""
    root = os.path.abspath(root)
    # Detect and resolve class root folder if necessary
    root = resolve_class_root(root)
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')
        ])
    else:
        class_names = list(class_names)
    if not class_names:
        raise ValueError(f"No class folders found under {root}")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    paths: List[str] = []
    labels_list: List[int] = []
    for name in class_names:
        class_dir = os.path.join(root, name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted([p for p in glob.glob(os.path.join(class_dir, '*')) if os.path.isfile(p)])
        paths.extend(files)
        labels_list.extend([class_to_idx[name]] * len(files))
    labels = np.array(labels_list, dtype=np.int32)
    if len(paths) == 0:
        raise ValueError(
            f"Found class folders under {root} but no image files. "
            "Ensure --teacher_data_root/--student_data_root points to the folder that directly contains class subfolders (e.g., .../Training)."
        )
    return paths, labels, class_names


class FruitDataset(Dataset):
    """Dataset that reads images from file paths and applies Albumentations transforms."""
    def __init__(self, paths: List[str], labels: np.ndarray, transform: Optional[A.Compose] = None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image_rgb)
            image = augmented['image']
        else:
            image = transforms.functional.to_tensor(image_rgb)
        label = int(self.labels[idx])
        return image, label


def save_history_json(history: Dict, path: str) -> None:
    """Persist metric history dictionaries for later analysis."""
    if not history:
        return
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def plot_metric_curves(metrics: Optional[Dict[str, List[float]]], title: str, save_path: str) -> None:
    """Plot train/validation loss and accuracy curves if data is available."""
    if not metrics or not metrics.get('train_loss'):
        return
    epochs = np.arange(1, len(metrics['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, metrics['train_loss'], label='Train', marker='o')
    val_loss = metrics.get('val_loss', [])
    if val_loss:
        axes[0].plot(epochs, val_loss, label='Val', marker='s')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, metrics.get('train_acc', []), label='Train', marker='o')
    val_acc = metrics.get('val_acc', [])
    if val_acc:
        axes[1].plot(epochs, val_acc, label='Val', marker='s')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, title: str) -> None:
    """Render and save a confusion matrix heatmap."""
    if cm.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j])), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
def build_resnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet50 model for classification."""
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vit(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ViT base model for classification."""
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
    except ImportError:
        raise ImportError("Vision Transformer models require torchvision ≥ 0.15")
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, correct / total


def predict_soft(model: nn.Module, images: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return softmax probabilities with temperature scaling."""
    with torch.no_grad():
        logits = model(images)
        logits = logits / temperature
        return torch.softmax(logits, dim=1)


def train_vit_with_distill(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    alpha: float,
    temperature: float,
) -> Dict[str, List[float]]:
    """
    Train the student ViT model with knowledge distillation from the teacher.

    Loss = (1 - alpha) * CE(student_logits, targets) + alpha * KD(student, teacher, T)
    where KD is KL divergence between teacher and student softmax outputs.
    """
    student.to(device)
    teacher.to(device)
    teacher.eval()  # freeze teacher
    optimizer = optim.AdamW(student.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            # Student forward
            outputs = student(images)
            # Teacher soft labels
            with torch.no_grad():
                teacher_probs = predict_soft(teacher, images, temperature)
            # Student log probabilities
            student_log_probs = torch.log_softmax(outputs / temperature, dim=1)
            # Losses
            loss_ce = ce_loss_fn(outputs, targets)
            loss_kd = kd_loss_fn(student_log_probs, teacher_probs) * (temperature ** 2)
            loss = (1 - alpha) * loss_ce + alpha * loss_kd
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(student, val_loader, ce_loss_fn, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"[Student] Epoch {epoch+1}/{epochs} - Train loss {train_loss:.4f} acc {train_acc:.4f} - Val loss {val_loss:.4f} acc {val_acc:.4f}")
    return history


# -----------------------------------------------------------------------------
# Teacher training pipeline (ResNet with warm‑up and refinement)
# -----------------------------------------------------------------------------
def train_teacher_pipeline(
    train_paths: List[str],
    train_labels: np.ndarray,
    val_paths: List[str],
    val_labels: np.ndarray,
    image_size: int,
    batch_size: int,
    warmup_epochs: int,
    refine_epochs: int,
    refine_margin: float,
    warmup_lr: float,
    refine_lr: float,
    warmup_freeze_base: bool,
    fine_tune_at: Optional[int],
    device: torch.device,
    output_dir: str,
    num_classes: int,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, Dict[str, Dict[str, List[float]]]]:
    """Train a ResNet teacher with warm-up and label refinement."""
    # Albumentations transforms
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.333), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    # Create datasets and loaders
    ds_train = FruitDataset(train_paths, train_labels, transform=train_transform)
    ds_val = FruitDataset(val_paths, val_labels, transform=eval_transform)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Build model
    model = build_resnet(num_classes, pretrained=True)
    model.to(device)
    history = {
        'warmup': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
        'refine': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
    }
    # Freeze base if requested during warm-up
    if warmup_freeze_base:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    # Warm‑up optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=warmup_lr)
    criterion = nn.CrossEntropyLoss()
    # Warm-up training
    for epoch in range(warmup_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history['warmup']['train_loss'].append(train_loss)
        history['warmup']['train_acc'].append(train_acc)
        history['warmup']['val_loss'].append(val_loss)
        history['warmup']['val_acc'].append(val_acc)
        print(f"[Teacher Warmup] Epoch {epoch+1}/{warmup_epochs} - Train loss {train_loss:.4f} acc {train_acc:.4f} - Val loss {val_loss:.4f} acc {val_acc:.4f}")
    # Label refinement
    ds_train_no_aug = FruitDataset(train_paths, train_labels, transform=eval_transform)
    loader_no_aug = DataLoader(ds_train_no_aug, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Compute predictions and confidences
    model.eval()
    preds_list = []
    conf_list = []
    with torch.no_grad():
        for images, _ in loader_no_aug:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = probs.max(dim=1)
            preds_list.append(preds.cpu().numpy())
            conf_list.append(conf.cpu().numpy())
    preds_all = np.concatenate(preds_list)
    conf_all = np.concatenate(conf_list)
    idx_refine = np.where((preds_all != train_labels) & (conf_all > refine_margin))[0]
    y_refined = train_labels.copy()
    y_refined[idx_refine] = preds_all[idx_refine]
    print(f"[Teacher] Refined {len(idx_refine)} labels out of {len(train_labels)} with margin {refine_margin}")
    # Unfreeze layers for refinement if specified
    for param in model.parameters():
        param.requires_grad = True
    # Optionally freeze early layers up to fine_tune_at index
    if fine_tune_at is not None and fine_tune_at >= 0:
        params = list(model.parameters())
        for i, param in enumerate(params):
            param.requires_grad = i >= fine_tune_at
    # New optimizer for refinement
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=refine_lr)
    # New dataset with refined labels
    ds_train_refined = FruitDataset(train_paths, y_refined, transform=train_transform)
    train_loader_refined = DataLoader(ds_train_refined, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Refinement training
    for epoch in range(refine_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader_refined, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history['refine']['train_loss'].append(train_loss)
        history['refine']['train_acc'].append(train_acc)
        history['refine']['val_loss'].append(val_loss)
        history['refine']['val_acc'].append(val_acc)
        print(f"[Teacher Refine] Epoch {epoch+1}/{refine_epochs} - Train loss {train_loss:.4f} acc {train_acc:.4f} - Val loss {val_loss:.4f} acc {val_acc:.4f}")
    # Save teacher model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'teacher_resnet.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[Teacher] Saved ResNet model to {model_path}")
    return model, y_refined, idx_refine, history


# -----------------------------------------------------------------------------
# Argument parsing and main execution
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fruit sorting with ResNet teacher and ViT student using knowledge distillation")
    # Teacher parameters
    parser.add_argument('--teacher_data_root', type=str, required=True, help='Root folder containing class subfolders for teacher training')
    parser.add_argument('--student_data_root', type=str, default=None, help='Root folder containing class subfolders for student training; defaults to teacher_data_root')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size for both teacher and student')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for both teacher and student')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Teacher noise and training
    parser.add_argument('--teacher_noise_mode', type=str, default='symmetric', choices=['symmetric','ccn','idn','mixed'], help='Noise mode for teacher')
    parser.add_argument('--teacher_noise_rate', type=float, default=0.2, help='Noise rate or max_rate for teacher')
    parser.add_argument('--teacher_ccn_P', type=str, default='', help='JSON confusion matrix for teacher class‑conditional noise')
    parser.add_argument('--teacher_mixed_cfg', type=str, default='', help='JSON config for teacher mixed noise')
    parser.add_argument('--teacher_warmup_epochs', type=int, default=6, help='Number of warm‑up epochs for teacher')
    parser.add_argument('--teacher_refine_epochs', type=int, default=6, help='Number of refinement epochs for teacher')
    parser.add_argument('--teacher_refine_margin', type=float, default=0.4, help='Confidence margin for label refinement')
    parser.add_argument('--teacher_warmup_lr', type=float, default=1e-4, help='Learning rate during teacher warm‑up')
    parser.add_argument('--teacher_refine_lr', type=float, default=5e-5, help='Learning rate during teacher refinement')
    parser.add_argument('--teacher_warmup_freeze_base', action='store_true', help='Freeze ResNet backbone during teacher warm‑up')
    parser.add_argument('--teacher_fine_tune_at', type=int, default=None, help='Parameter index to start unfreezing during teacher refinement')
    parser.add_argument('--teacher_val_split', type=float, default=0.1, help='Validation split fraction for teacher when no val folder exists')
    # Student noise and training
    parser.add_argument('--student_noise_mode', type=str, default='symmetric', choices=['symmetric','ccn','idn','mixed'], help='Noise mode for student')
    parser.add_argument('--student_noise_rate', type=float, default=0.0, help='Noise rate or max_rate for student')
    parser.add_argument('--student_ccn_P', type=str, default='', help='JSON confusion matrix for student class‑conditional noise')
    parser.add_argument('--student_mixed_cfg', type=str, default='', help='JSON config for student mixed noise')
    parser.add_argument('--student_epochs', type=int, default=10, help='Number of training epochs for the student ViT')
    parser.add_argument('--student_lr', type=float, default=3e-4, help='Learning rate for the student')
    parser.add_argument('--distill_alpha', type=float, default=0.5, help='Weight for distillation loss')
    parser.add_argument('--distill_temperature', type=float, default=2.0, help='Temperature for distillation')
    parser.add_argument('--student_val_split', type=float, default=0.1, help='Validation split fraction for student when no val folder exists')
    # Output
    parser.add_argument('--output_dir', type=str, default='runs', help='Directory to save outputs')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively if possible')
    return parser.parse_args()


def log_dataset_summary(name: str, paths: List[str], labels: np.ndarray, class_names: List[str]) -> None:
    """Print quick stats so users can verify dataset discovery."""
    unique, counts = np.unique(labels, return_counts=True)
    sample_pairs = sorted(zip(unique, counts), key=lambda x: int(x[1]), reverse=True)
    head = ', '.join(f"{class_names[idx]}:{count}" for idx, count in sample_pairs[:5])
    print(f"[{name}] {len(paths)} images across {len(unique)}/{len(class_names)} classes. Top counts: {head}")


def ensure_stratified_ready(labels: np.ndarray, class_names: List[str], prefix: str) -> None:
    """Raise a clear error when stratified split inputs are empty or too small."""
    if labels.size == 0:
        raise ValueError(f"[{prefix}] Cannot stratify because no samples were found.")
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        raise ValueError(f"[{prefix}] Stratified split needs at least two classes but only found {len(unique)}.")
    scarce = [class_names[int(u)] for u, c in zip(unique, counts) if c < 2]
    if scarce:
        raise ValueError(
            f"[{prefix}] Stratified split requires ≥2 samples per class. These classes are undersized: {', '.join(scarce)}."
        )


def main() -> None:
    args = parse_args()
    setup_matplotlib(args.show_plots)
    os.makedirs(args.output_dir, exist_ok=True)
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Teacher dataset path
    teacher_root_arg = os.path.abspath(args.teacher_data_root)
    teacher_root = resolve_class_root(teacher_root_arg)
    student_root_arg = os.path.abspath(args.student_data_root) if args.student_data_root else teacher_root_arg
    student_root = resolve_class_root(student_root_arg)
    # Detect class names from teacher dataset
    class_dirs = [d for d in os.listdir(teacher_root) if os.path.isdir(os.path.join(teacher_root, d))]
    class_names = sorted([d for d in class_dirs if not d.startswith('.')])
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("The dataset must contain at least two classes")
    # Load teacher dataset
    teacher_paths, y_teacher_clean, class_names = list_images_with_labels(teacher_root, class_names)
    log_dataset_summary("Teacher", teacher_paths, y_teacher_clean, class_names)
    # Determine teacher validation set
    val_dir = None
    # If there is a subdirectory named 'val' or 'validation', use it; else split
    for dname in ['val','validation']:
        candidate = os.path.join(os.path.dirname(teacher_root), dname)
        if os.path.isdir(candidate):
            val_dir = candidate
            break
    if val_dir:
        val_paths, y_val, _ = list_images_with_labels(val_dir, class_names)
    else:
        # Stratified split
        from sklearn.model_selection import StratifiedShuffleSplit
        ensure_stratified_ready(y_teacher_clean, class_names, "Teacher")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.teacher_val_split, random_state=args.seed)
        teacher_paths_np = np.array(teacher_paths)
        y_teacher_np = np.array(y_teacher_clean)
        train_idx, val_idx = next(sss.split(teacher_paths_np, y_teacher_np))
        val_paths = teacher_paths_np[val_idx].tolist()
        y_val = y_teacher_np[val_idx]
        teacher_paths = teacher_paths_np[train_idx].tolist()
        y_teacher_clean = y_teacher_np[train_idx]
        print(f"[Teacher] Created validation split of size {len(val_paths)}")
    # Compute difficulty metrics for teacher dataset
    lap_teacher, ent_teacher = compute_difficulty_metrics(teacher_paths, args.image_size)
    # Inject noise into teacher labels
    rng = np.random.default_rng(args.seed)
    if args.teacher_noise_mode == 'symmetric':
        y_teacher_noisy, idx_teacher_noise = inject_symmetric_noise(y_teacher_clean, args.teacher_noise_rate, rng, num_classes)
    elif args.teacher_noise_mode == 'ccn':
        if not args.teacher_ccn_P:
            raise ValueError("Provide --teacher_ccn_P JSON for class‑conditional noise")
        P = np.array(json.loads(args.teacher_ccn_P), dtype=np.float32)
        if P.shape[0] != num_classes or P.shape[1] != num_classes:
            raise ValueError("Confusion matrix dimensions mismatch for teacher")
        y_teacher_noisy, idx_teacher_noise = inject_ccn_noise(y_teacher_clean, P, rng)
    elif args.teacher_noise_mode == 'idn':
        y_teacher_noisy, idx_teacher_noise = inject_instance_dep_noise(y_teacher_clean, lap_teacher, ent_teacher, args.teacher_noise_rate, rng, num_classes)
    elif args.teacher_noise_mode == 'mixed':
        if not args.teacher_mixed_cfg:
            raise ValueError("Provide --teacher_mixed_cfg JSON for mixed noise")
        cfg = json.loads(args.teacher_mixed_cfg)
        y_teacher_noisy, idx_teacher_noise = inject_mixed_noise(y_teacher_clean, lap_teacher, ent_teacher, cfg, rng, num_classes)
    else:
        raise ValueError("Invalid teacher noise mode")
    print(f"[Teacher] Injected noise: {len(idx_teacher_noise)} labels flipped out of {len(y_teacher_clean)}")
    # Train teacher
    teacher_model, y_teacher_refined, idx_refine, teacher_history = train_teacher_pipeline(
        train_paths=teacher_paths,
        train_labels=y_teacher_noisy,
        val_paths=val_paths,
        val_labels=y_val,
        image_size=args.image_size,
        batch_size=args.batch_size,
        warmup_epochs=args.teacher_warmup_epochs,
        refine_epochs=args.teacher_refine_epochs,
        refine_margin=args.teacher_refine_margin,
        warmup_lr=args.teacher_warmup_lr,
        refine_lr=args.teacher_refine_lr,
        warmup_freeze_base=args.teacher_warmup_freeze_base,
        fine_tune_at=args.teacher_fine_tune_at,
        device=device,
        output_dir=args.output_dir,
        num_classes=num_classes,
    )
    # Student dataset
    student_class_names = class_names  # assume same class order
    student_paths, y_student_clean, _ = list_images_with_labels(student_root, student_class_names)
    log_dataset_summary("Student", student_paths, y_student_clean, student_class_names)
    # Determine student validation set
    student_val_dir = None
    for dname in ['val','validation']:
        candidate = os.path.join(os.path.dirname(student_root), dname)
        if os.path.isdir(candidate):
            student_val_dir = candidate
            break
    if student_val_dir:
        student_val_paths, y_student_val, _ = list_images_with_labels(student_val_dir, student_class_names)
    else:
        # Stratified split
        from sklearn.model_selection import StratifiedShuffleSplit
        ensure_stratified_ready(y_student_clean, student_class_names, "Student")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.student_val_split, random_state=args.seed)
        student_paths_np = np.array(student_paths)
        y_student_np = np.array(y_student_clean)
        train_idx, val_idx = next(sss.split(student_paths_np, y_student_np))
        student_val_paths = student_paths_np[val_idx].tolist()
        y_student_val = y_student_np[val_idx]
        student_paths = student_paths_np[train_idx].tolist()
        y_student_clean = y_student_np[train_idx]
        print(f"[Student] Created validation split of size {len(student_val_paths)}")
    # Compute difficulty metrics for student dataset
    lap_student, ent_student = compute_difficulty_metrics(student_paths, args.image_size)
    # Inject noise into student labels
    rng2 = np.random.default_rng(args.seed)
    if args.student_noise_mode == 'symmetric':
        y_student_noisy, idx_student_noise = inject_symmetric_noise(y_student_clean, args.student_noise_rate, rng2, num_classes)
    elif args.student_noise_mode == 'ccn':
        if not args.student_ccn_P:
            raise ValueError("Provide --student_ccn_P JSON for class‑conditional noise")
        P = np.array(json.loads(args.student_ccn_P), dtype=np.float32)
        if P.shape[0] != num_classes or P.shape[1] != num_classes:
            raise ValueError("Confusion matrix dimensions mismatch for student")
        y_student_noisy, idx_student_noise = inject_ccn_noise(y_student_clean, P, rng2)
    elif args.student_noise_mode == 'idn':
        y_student_noisy, idx_student_noise = inject_instance_dep_noise(y_student_clean, lap_student, ent_student, args.student_noise_rate, rng2, num_classes)
    elif args.student_noise_mode == 'mixed':
        if not args.student_mixed_cfg:
            raise ValueError("Provide --student_mixed_cfg JSON for mixed noise")
        cfg = json.loads(args.student_mixed_cfg)
        y_student_noisy, idx_student_noise = inject_mixed_noise(y_student_clean, lap_student, ent_student, cfg, rng2, num_classes)
    else:
        raise ValueError("Invalid student noise mode")
    print(f"[Student] Injected noise: {len(idx_student_noise)} labels flipped out of {len(y_student_clean)}")
    # Create student transforms
    student_train_transform = A.Compose([
        A.RandomResizedCrop(size=(args.image_size, args.image_size), scale=(0.8, 1.0), ratio=(0.75, 1.333), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    student_eval_transform = A.Compose([
        A.Resize(height=args.image_size, width=args.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    # Create student datasets and loaders
    ds_student_train = FruitDataset(student_paths, y_student_noisy, transform=student_train_transform)
    ds_student_val = FruitDataset(student_val_paths, y_student_val, transform=student_eval_transform)
    train_loader_student = DataLoader(ds_student_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_student = DataLoader(ds_student_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Build student model
    student_model = build_vit(num_classes, pretrained=True)
    # Train student with distillation
    student_history = train_vit_with_distill(
        student=student_model,
        teacher=teacher_model,
        train_loader=train_loader_student,
        val_loader=val_loader_student,
        device=device,
        epochs=args.student_epochs,
        lr=args.student_lr,
        alpha=args.distill_alpha,
        temperature=args.distill_temperature,
    )
    # Save student model
    student_path = os.path.join(args.output_dir, 'student_vit.pth')
    torch.save(student_model.state_dict(), student_path)
    print(f"[Student] Saved ViT model to {student_path}")
    # Evaluate student on its validation set
    ce_loss_fn = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(student_model, val_loader_student, ce_loss_fn, device)
    print(f"[Student] Final validation loss {val_loss:.4f} acc {val_acc:.4f}")
    # Optionally evaluate on teacher's test set if student_root == teacher_root
    if student_root == teacher_root:
        ds_test = FruitDataset(val_paths, y_val, transform=student_eval_transform)
        test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loss, test_acc = evaluate(student_model, test_loader, ce_loss_fn, device)
        print(f"[Student] Test loss {test_loss:.4f} acc {test_acc:.4f}")
    # Save confusion matrix for student
    # Compute predictions on student validation set
    preds_all = []
    true_all = []
    student_model.eval()
    with torch.no_grad():
        for images, targets in val_loader_student:
            images = images.to(device)
            outputs = student_model(images)
            preds = outputs.argmax(dim=1)
            preds_all.append(preds.cpu().numpy())
            true_all.append(targets.numpy())
    preds_all = np.concatenate(preds_all)
    true_all = np.concatenate(true_all)
    cm = confusion_matrix(true_all, preds_all, labels=np.arange(num_classes))
    cm_path = os.path.join(args.output_dir, 'student_confusion_matrix.npy')
    np.save(cm_path, cm)
    print(f"[Student] Confusion matrix saved to {cm_path}")
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_history_json(teacher_history, os.path.join(args.output_dir, 'teacher_history.json'))
    save_history_json(student_history, os.path.join(args.output_dir, 'student_history.json'))
    plot_metric_curves(teacher_history.get('warmup'), 'Teacher Warm-up Metrics', os.path.join(plots_dir, 'teacher_warmup_curves.png'))
    plot_metric_curves(teacher_history.get('refine'), 'Teacher Refinement Metrics', os.path.join(plots_dir, 'teacher_refine_curves.png'))
    plot_metric_curves(student_history, 'Student KD Metrics', os.path.join(plots_dir, 'student_distillation_curves.png'))
    plot_confusion_matrix(cm, class_names, os.path.join(plots_dir, 'student_confusion_matrix_heatmap.png'), title='Student Confusion Matrix (Validation)')
    # Optionally show plots interactively
    if args.show_plots and plt is not None:
        try:
            plt.show()
        except Exception as exc:
            print(f"[Plot] Failed to show plots interactively: {exc}")

if __name__ == "__main__":
    main()