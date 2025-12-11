import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(dedent("""
# IFT3395 Competition 2 - Improved Deep MLP + Advanced Regularization

改进策略：
- 2-3 层隐藏层的深层 MLP
- 更强的正则化（高 dropout + 强 L2）
- 更激进的类别权重
- 更精细的超参搜索
- 更长的训练时间 + 早停
- 增强的数据增强策略
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path('data')
SEED = 3030
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
class_weights_aggressive = class_weights_base ** 1.5
class_weights_aggressive = class_weights_aggressive / class_weights_aggressive.mean()
print('Class counts:', dict(zip(unique, counts)))
print('Aggressive class weights:', class_weights_aggressive)
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
def augment_batch_advanced(raw_batch: np.ndarray, p_flip: float = 0.5, noise_std: float = 0.025, brightness: float = 0.12, contrast: float = 0.1) -> np.ndarray:
    imgs = raw_batch.reshape(len(raw_batch), 28, 28, 3).copy()
    if np.random.rand() < p_flip:
        imgs = imgs[:, :, ::-1, :]
    if np.random.rand() < p_flip * 0.5:
        imgs = imgs[:, ::-1, :, :]
    imgs += np.random.normal(0.0, noise_std, size=imgs.shape)
    brightness_shift = (np.random.rand() - 0.5) * brightness
    imgs += brightness_shift
    contrast_factor = 1.0 + (np.random.rand() - 0.5) * contrast
    imgs = (imgs - 0.5) * contrast_factor + 0.5
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
        aug = np.clip(imgs + 0.06, 0.0, 1.0)
    elif mode == 'bright-':
        aug = np.clip(imgs - 0.06, 0.0, 1.0)
    else:
        aug = imgs
    return aug.reshape(len(raw_batch), -1)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class DeepMLP:
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int], lr: float, l2: float, dropout: float, seed: int = 0):
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.lr = lr
        self.base_lr = lr
        self.l2 = l2
        self.dropout = dropout
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            self.weights.append(rng.normal(0.0, np.sqrt(2.0 / dims[i]), size=(dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1], dtype=np.float32))

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, x: np.ndarray, train: bool = False) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        cache = []
        h = x
        for i in range(self.num_layers):
            if train and self.dropout > 0.0:
                mask = (np.random.random(size=h.shape) >= self.dropout).astype(np.float32)
                h = h * mask
            z = h @ self.weights[i] + self.biases[i]
            if i < self.num_layers - 1:
                h = np.maximum(0.0, z)
            else:
                h = z
            cache.append((h.copy() if i < self.num_layers - 1 else None, z))
        return h, cache

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray, class_weights: np.ndarray) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        logits, cache = self._forward(x, train=True)
        probs = self._softmax(logits)
        batch = x.shape[0]
        sample_w = class_weights[y]
        loss = -np.sum(sample_w * np.log(probs[np.arange(batch), y] + 1e-8)) / sample_w.sum()
        for w in self.weights:
            loss += 0.5 * self.l2 * np.sum(w * w)
        grad_logits = probs
        grad_logits[np.arange(batch), y] -= 1.0
        grad_logits *= (sample_w[:, None] / sample_w.sum())
        grad_w_list = []
        grad_b_list = []
        grad = grad_logits
        for i in range(self.num_layers - 1, -1, -1):
            h_prev = cache[i - 1][0] if i > 0 else x
            grad_w = h_prev.T @ grad + self.l2 * self.weights[i]
            grad_b = grad.sum(axis=0)
            grad_w_list.insert(0, grad_w)
            grad_b_list.insert(0, grad_b)
            if i > 0:
                grad = grad @ self.weights[i].T
                z_prev = cache[i - 1][1]
                grad[z_prev <= 0.0] = 0.0
        return loss, grad_w_list, grad_b_list

    def fit(self, x: np.ndarray, y: np.ndarray, *, class_weights: np.ndarray, epochs: int, batch_size: int, val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, raw_data: Optional[np.ndarray] = None, augment: bool = False, patience: int = 50) -> Dict[str, List[float]]:
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
                    raw_aug = augment_batch_advanced(raw_data[idx])
                    xb = standardize_raw(raw_aug)
                yb = y[idx]
                _, grad_w_list, grad_b_list = self._loss_and_grads(xb, yb, class_weights)
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grad_w_list[i]
                    self.biases[i] -= self.lr * grad_b_list[i]
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
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
        }

    def load_state(self, state):
        self.weights = [w.copy() for w in state['weights']]
        self.biases = [b.copy() for b in state['biases']]
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
    {'hidden_dims': [512, 256], 'lr': 0.04, 'l2': 1.2e-4, 'dropout': 0.25, 'epochs': 400, 'batch_size': 96, 'patience': 60},
    {'hidden_dims': [640, 320], 'lr': 0.045, 'l2': 1.5e-4, 'dropout': 0.3, 'epochs': 450, 'batch_size': 80, 'patience': 70},
    {'hidden_dims': [768, 384], 'lr': 0.035, 'l2': 1e-4, 'dropout': 0.2, 'epochs': 500, 'batch_size': 112, 'patience': 80},
    {'hidden_dims': [512, 256, 128], 'lr': 0.038, 'l2': 1.3e-4, 'dropout': 0.28, 'epochs': 420, 'batch_size': 88, 'patience': 65},
    {'hidden_dims': [640, 320, 160], 'lr': 0.042, 'l2': 1.4e-4, 'dropout': 0.22, 'epochs': 480, 'batch_size': 96, 'patience': 75},
]

results = []
for i, params in enumerate(search_space, 1):
    print(f"Config {i}/{len(search_space)}: {params}")
    model = DeepMLP(
        input_dim=x_train_std.shape[1],
        num_classes=num_classes,
        hidden_dims=params['hidden_dims'],
        lr=params['lr'],
        l2=params['l2'],
        dropout=params['dropout'],
        seed=SEED + i
    )
    history = model.fit(
        x_train_std,
        y_train,
        class_weights=class_weights_aggressive,
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
models: List[DeepMLP] = []
for idx, cfg in enumerate(sorted_configs):
    cfg_full = cfg.copy()
    cfg_full['epochs'] = cfg['epochs'] + 60
    cfg_full['patience'] = cfg['patience'] + 20
    print(f"Retraining top config {idx + 1}: hidden_dims={cfg_full['hidden_dims']}, lr={cfg_full['lr']:.4f}, l2={cfg_full['l2']:.2e}, dropout={cfg_full['dropout']:.2f}")
    model = DeepMLP(
        input_dim=x_std.shape[1],
        num_classes=num_classes,
        hidden_dims=cfg_full['hidden_dims'],
        lr=cfg_full['lr'],
        l2=cfg_full['l2'],
        dropout=cfg_full['dropout'],
        seed=SEED * 3 + idx
    )
    train_idx_full, val_idx_full = train_val_split_indices(len(x_std), val_ratio=0.1, seed=SEED * 4 + idx)
    model.fit(
        x_std[train_idx_full],
        labels[train_idx_full],
        class_weights=class_weights_aggressive,
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
submission_path = Path('submission_improved_deep.csv')
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

nbf.write(nb, 'ift3395_improved_deep.ipynb')
print('Notebook created: ift3395_improved_deep.ipynb')
