import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(dedent("""
# IFT3395 Competition 2 - Robust Wide MLP (Optimized)

稳健策略：
- 使用更宽的2层网络（避免梯度问题）
- 平衡的正则化（适中 dropout + L2）
- He 初始化（更好的梯度流）
- 梯度裁剪防止爆炸
- 更精细的超参搜索
- 数据增强 + TTA
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path('data')
SEED = 4040
rng = np.random.default_rng(SEED)
np.set_printoptions(precision=4, suppress=True)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def load_split(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"{split}_data.pkl"
    with path.open('rb') as f:
        return pickle.load(f)

train_data = load_split('train')
test_data = load_split('test')

images = train_data['images'].astype(np.float32)
labels = train_data['labels'].reshape(-1).astype(int)
test_images = test_data['images'].astype(np.float32)
num_classes = len(np.unique(labels))
print(f"Train: {images.shape}, Test: {test_images.shape}, Classes: {num_classes}")
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
unique, counts = np.unique(labels, return_counts=True)
class_weights_base = counts.sum() / (counts.astype(np.float32) + 1e-6)
class_weights = class_weights_base ** 1.2
class_weights = class_weights / class_weights.mean()
print('Class counts:', dict(zip(unique, counts)))
print('Class weights:', class_weights)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def flatten_and_normalize(imgs: np.ndarray) -> np.ndarray:
    flat = imgs.reshape(len(imgs), -1).astype(np.float32)
    return flat / 255.0

class StandardScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> 'StandardScaler':
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-5
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError('Scaler not fitted')
        return (x - self.mean_) / self.std_

x_norm = flatten_and_normalize(images)
x_test_norm = flatten_and_normalize(test_images)
scaler = StandardScaler().fit(x_norm)
x_std = scaler.transform(x_norm)
x_test_std = scaler.transform(x_test_norm)
SCALE_MEAN = scaler.mean_
SCALE_STD = scaler.std_


def standardize_raw(raw_batch: np.ndarray) -> np.ndarray:
    return (raw_batch - SCALE_MEAN) / SCALE_STD


def train_val_split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_size = int(n * val_ratio)
    return idx[val_size:], idx[:val_size]
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def augment_batch(raw_batch: np.ndarray, p_flip: float = 0.5, noise_std: float = 0.02, brightness: float = 0.1) -> np.ndarray:
    imgs = raw_batch.reshape(len(raw_batch), 28, 28, 3).copy()
    if np.random.rand() < p_flip:
        imgs = imgs[:, :, ::-1, :]
    if np.random.rand() < p_flip * 0.5:
        imgs = imgs[:, ::-1, :, :]
    imgs += np.random.normal(0.0, noise_std, size=imgs.shape)
    imgs += (np.random.rand() - 0.5) * brightness
    imgs = np.clip(imgs, 0.0, 1.0)
    return imgs.reshape(len(raw_batch), -1)

TTA_MODES = ['identity', 'hflip', 'vflip', 'bright+', 'bright-']


def apply_tta(raw_batch: np.ndarray, mode: str) -> np.ndarray:
    imgs = raw_batch.reshape(len(raw_batch), 28, 28, 3)
    if mode == 'hflip':
        aug = imgs[:, :, ::-1, :]
    elif mode == 'vflip':
        aug = imgs[:, ::-1, :, :]
    elif mode == 'bright+':
        aug = np.clip(imgs + 0.05, 0.0, 1.0)
    elif mode == 'bright-':
        aug = np.clip(imgs - 0.05, 0.0, 1.0)
    else:
        aug = imgs
    return aug.reshape(len(raw_batch), -1)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class RobustMLP:
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, lr: float, l2: float, dropout: float, seed: int = 0, clip_grad: float = 5.0):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.base_lr = lr
        self.l2 = l2
        self.dropout = dropout
        self.clip_grad = clip_grad
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w1 = rng.normal(0.0, scale1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, scale2, size=(hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _clip_gradients(self, grad_w1: np.ndarray, grad_b1: np.ndarray, grad_w2: np.ndarray, grad_b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        norm = np.sqrt(np.sum(grad_w1 * grad_w1) + np.sum(grad_b1 * grad_b1) + np.sum(grad_w2 * grad_w2) + np.sum(grad_b2 * grad_b2))
        if norm > self.clip_grad:
            scale = self.clip_grad / norm
            grad_w1 = grad_w1 * scale
            grad_b1 = grad_b1 * scale
            grad_w2 = grad_w2 * scale
            grad_b2 = grad_b2 * scale
        return grad_w1, grad_b1, grad_w2, grad_b2

    def _forward(self, x: np.ndarray, train: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if train and self.dropout > 0.0:
            mask = (np.random.random(size=x.shape) >= self.dropout).astype(np.float32)
            x = x * mask
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0.0, z1)
        if train and self.dropout > 0.0:
            mask_h = (np.random.random(size=h1.shape) >= self.dropout).astype(np.float32)
            h1 = h1 * mask_h
        logits = h1 @ self.w2 + self.b2
        return logits, (x, z1, h1)

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray, class_weights: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logits, cache = self._forward(x, train=True)
        probs = self._softmax(logits)
        batch = x.shape[0]
        sample_w = class_weights[y]
        loss = -np.sum(sample_w * np.log(probs[np.arange(batch), y] + 1e-8)) / sample_w.sum()
        loss += 0.5 * self.l2 * (np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2))
        grad_logits = probs
        grad_logits[np.arange(batch), y] -= 1.0
        grad_logits *= (sample_w[:, None] / sample_w.sum())
        x_cache, z1, h1 = cache
        grad_w2 = h1.T @ grad_logits + self.l2 * self.w2
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.w2.T
        grad_hidden[z1 <= 0.0] = 0.0
        grad_w1 = x_cache.T @ grad_hidden + self.l2 * self.w1
        grad_b1 = grad_hidden.sum(axis=0)
        grad_w1, grad_b1, grad_w2, grad_b2 = self._clip_gradients(grad_w1, grad_b1, grad_w2, grad_b2)
        return loss, grad_w1, grad_b1, grad_w2, grad_b2

    def fit(self, x: np.ndarray, y: np.ndarray, *, class_weights: np.ndarray, epochs: int, batch_size: int, val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, raw_data: Optional[np.ndarray] = None, augment: bool = False, patience: int = 40) -> Dict[str, List[float]]:
        history = {'train_acc': [], 'val_acc': []}
        best_state = self.get_state()
        best_val = -np.inf
        wait = 0
        num_samples = x.shape[0]
        for epoch in range(epochs):
            lr_scale = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            self.lr = self.base_lr * lr_scale
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                xb = x[idx]
                if augment and raw_data is not None:
                    raw_aug = augment_batch(raw_data[idx])
                    xb = standardize_raw(raw_aug)
                yb = y[idx]
                _, grad_w1, grad_b1, grad_w2, grad_b2 = self._loss_and_grads(xb, yb, class_weights)
                self.w1 -= self.lr * grad_w1
                self.b1 -= self.lr * grad_b1
                self.w2 -= self.lr * grad_w2
                self.b2 -= self.lr * grad_b2
            train_acc = self.accuracy(x, y)
            history['train_acc'].append(train_acc)
            if val_data is not None:
                val_acc = self.accuracy(*val_data)
                history['val_acc'].append(val_acc)
                if val_acc > best_val + 1e-4:
                    best_val = val_acc
                    best_state = self.get_state()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if val_data is not None:
            self.load_state(best_state)
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x, train=False)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())

    def get_state(self):
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'w2': self.w2.copy(),
            'b2': self.b2.copy(),
        }

    def load_state(self, state):
        self.w1 = state['w1'].copy()
        self.b1 = state['b1'].copy()
        self.w2 = state['w2'].copy()
        self.b2 = state['b2'].copy()
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
train_idx, val_idx = train_val_split_indices(len(x_std), val_ratio=0.2, seed=SEED)
x_train_std = x_std[train_idx]
x_val_std = x_std[val_idx]
x_train_raw = x_norm[train_idx]
x_val_raw = x_norm[val_idx]
y_train = labels[train_idx]
y_val = labels[val_idx]
print(f"Train: {x_train_std.shape}, Val: {x_val_std.shape}")
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
search_space = [
    {'hidden_dim': 512, 'lr': 0.05, 'l2': 6e-5, 'dropout': 0.15, 'epochs': 350, 'batch_size': 112, 'patience': 50},
    {'hidden_dim': 640, 'lr': 0.055, 'l2': 8e-5, 'dropout': 0.18, 'epochs': 380, 'batch_size': 96, 'patience': 55},
    {'hidden_dim': 768, 'lr': 0.048, 'l2': 7e-5, 'dropout': 0.12, 'epochs': 400, 'batch_size': 128, 'patience': 60},
    {'hidden_dim': 896, 'lr': 0.052, 'l2': 9e-5, 'dropout': 0.2, 'epochs': 360, 'batch_size': 104, 'patience': 50},
    {'hidden_dim': 640, 'lr': 0.045, 'l2': 5e-5, 'dropout': 0.1, 'epochs': 420, 'batch_size': 120, 'patience': 65},
]

results = []
for i, params in enumerate(search_space, 1):
    print(f"Config {i}/{len(search_space)}: hidden_dim={params['hidden_dim']}, lr={params['lr']:.3f}, l2={params['l2']:.2e}, dropout={params['dropout']:.2f}")
    model = RobustMLP(
        input_dim=x_train_std.shape[1],
        num_classes=num_classes,
        hidden_dim=params['hidden_dim'],
        lr=params['lr'],
        l2=params['l2'],
        dropout=params['dropout'],
        seed=SEED + i
    )
    history = model.fit(
        x_train_std,
        y_train,
        class_weights=class_weights,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        val_data=(x_val_std, y_val),
        raw_data=x_train_raw,
        augment=True,
        patience=params['patience']
    )
    train_acc = history['train_acc'][-1]
    val_acc = history['val_acc'][-1]
    results.append({**params, 'train_acc': train_acc, 'val_acc': val_acc})
    print(f"  train acc={train_acc:.4f}, val acc={val_acc:.4f}\n")

best = max(results, key=lambda r: r['val_acc'])
print('Best config:', {k: v for k, v in best.items() if k not in {'train_acc', 'val_acc'}})
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
TOP_K = 3
sorted_configs = sorted(results, key=lambda r: r['val_acc'], reverse=True)[:TOP_K]
models: List[RobustMLP] = []
for idx, cfg in enumerate(sorted_configs):
    cfg_full = cfg.copy()
    cfg_full['epochs'] = cfg['epochs'] + 50
    cfg_full['patience'] = cfg['patience'] + 15
    print(f"Retraining top config {idx + 1}: hidden_dim={cfg_full['hidden_dim']}, lr={cfg_full['lr']:.3f}, l2={cfg_full['l2']:.2e}, dropout={cfg_full['dropout']:.2f}")
    model = RobustMLP(
        input_dim=x_std.shape[1],
        num_classes=num_classes,
        hidden_dim=cfg_full['hidden_dim'],
        lr=cfg_full['lr'],
        l2=cfg_full['l2'],
        dropout=cfg_full['dropout'],
        seed=SEED * 5 + idx
    )
    train_idx_full, val_idx_full = train_val_split_indices(len(x_std), val_ratio=0.1, seed=SEED * 6 + idx)
    model.fit(
        x_std[train_idx_full],
        labels[train_idx_full],
        class_weights=class_weights,
        epochs=cfg_full['epochs'],
        batch_size=cfg_full['batch_size'],
        val_data=(x_std[val_idx_full], labels[val_idx_full]),
        raw_data=x_norm[train_idx_full],
        augment=True,
        patience=cfg_full['patience']
    )
    models.append(model)
print(f"Total models trained: {len(models)}")
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
probs_accum = np.zeros((len(x_test_std), num_classes), dtype=np.float32)
for model in models:
    for mode in TTA_MODES:
        raw_aug = apply_tta(x_test_norm, mode)
        std_aug = standardize_raw(raw_aug)
        probs_accum += model.predict_proba(std_aug)

probs_accum /= (len(models) * len(TTA_MODES))
test_preds = np.argmax(probs_accum, axis=1)

ids = [str(i) for i in range(1, len(test_preds) + 1)]
submission_path = Path('submission_robust_wide.csv')
with submission_path.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Label'])
    for idx, label in zip(ids, test_preds):
        writer.writerow([idx, int(label)])

print(f'Submission saved to {submission_path.resolve()}')
""")))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'file_extension': '.py'
    }
}

nbf.write(nb, 'ift3395_robust_wide.ipynb')
print('Notebook created: ift3395_robust_wide.ipynb')
