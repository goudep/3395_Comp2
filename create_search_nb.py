import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(dedent("""
# IFT3395 Competition 2 - Hyperparameter Search (NumPy)

目的：在只使用 NumPy 的前提下，对 MLP 模型进行系统化超参搜索，选择验证表现最好的配置，然后用该配置在全量数据上训练并生成提交文件。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path('data')
SEED = 2025
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
class_weights = counts.sum() / (counts.astype(np.float32) + 1e-6)
class_weights = class_weights / class_weights.mean()
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

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

def train_val_split(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    val_size = int(len(x) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]

x_flat = flatten_and_normalize(images)
x_test_flat = flatten_and_normalize(test_images)
scaler = StandardScaler().fit(x_flat)
x_std = scaler.transform(x_flat)
x_test_std = scaler.transform(x_test_flat)
x_train, x_val, y_train, y_val = train_val_split(x_std, labels, val_ratio=0.2, seed=SEED)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class NeuralNetClassifier:
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, lr: float, l2: float, dropout: float, seed: int = 0):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.base_lr = lr
        self.l2 = l2
        self.dropout = dropout
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.02, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.02, size=(hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, x: np.ndarray, train: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if train and self.dropout > 0.0:
            mask = (np.random.random(size=x.shape) >= self.dropout).astype(np.float32)
            x = x * mask
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0.0, z1)
        logits = h1 @ self.w2 + self.b2
        return logits, (x, z1, h1)

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray, class_weights: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
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
        grads: Dict[str, np.ndarray] = {}
        grads['w2'] = h1.T @ grad_logits + self.l2 * self.w2
        grads['b2'] = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.w2.T
        grad_hidden[z1 <= 0.0] = 0.0
        grads['w1'] = x_cache.T @ grad_hidden + self.l2 * self.w1
        grads['b1'] = grad_hidden.sum(axis=0)
        return loss, grads

    def fit(self, x: np.ndarray, y: np.ndarray, *, class_weights: np.ndarray, epochs: int, batch_size: int, val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {'train_acc': [], 'val_acc': []}
        num_samples = x.shape[0]
        for epoch in range(epochs):
            lr_scale = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            self.lr = self.base_lr * lr_scale
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                xb = x[batch_idx]
                yb = y[batch_idx]
                _, grads = self._loss_and_grads(xb, yb, class_weights)
                self.w1 -= self.lr * grads['w1']
                self.b1 -= self.lr * grads['b1']
                self.w2 -= self.lr * grads['w2']
                self.b2 -= self.lr * grads['b2']
            history['train_acc'].append(self.accuracy(x, y))
            if val_data is not None:
                history['val_acc'].append(self.accuracy(*val_data))
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x, train=False)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
search_space = [
    {'hidden_dim': 384, 'lr': 0.05, 'l2': 5e-5, 'dropout': 0.05, 'epochs': 240, 'batch_size': 128},
    {'hidden_dim': 512, 'lr': 0.06, 'l2': 8e-5, 'dropout': 0.1, 'epochs': 280, 'batch_size': 96},
    {'hidden_dim': 640, 'lr': 0.07, 'l2': 1e-4, 'dropout': 0.15, 'epochs': 300, 'batch_size': 96},
    {'hidden_dim': 768, 'lr': 0.045, 'l2': 6e-5, 'dropout': 0.1, 'epochs': 320, 'batch_size': 128},
    {'hidden_dim': 512, 'lr': 0.055, 'l2': 3e-5, 'dropout': 0.08, 'epochs': 260, 'batch_size': 112},
]

results = []
for i, params in enumerate(search_space, 1):
    print(f"Config {i}/{len(search_space)}: {params}")
    model = NeuralNetClassifier(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        hidden_dim=params['hidden_dim'],
        lr=params['lr'],
        l2=params['l2'],
        dropout=params['dropout'],
        seed=SEED + i
    )
    history = model.fit(
        x_train,
        y_train,
        class_weights=class_weights,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        val_data=(x_val, y_val)
    )
    val_acc = history['val_acc'][-1]
    train_acc = history['train_acc'][-1]
    results.append({**params, 'train_acc': train_acc, 'val_acc': val_acc, 'model': model})
    print(f"  train acc={train_acc:.4f}, val acc={val_acc:.4f}\n")

best = max(results, key=lambda r: r['val_acc'])
print('Best config:', {k: v for k, v in best.items() if k not in {'model'}})
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
full_model = NeuralNetClassifier(
    input_dim=x_std.shape[1],
    num_classes=num_classes,
    hidden_dim=best['hidden_dim'],
    lr=best['lr'],
    l2=best['l2'],
    dropout=best['dropout'],
    seed=SEED
)
full_model.fit(x_std, labels, class_weights=class_weights, epochs=best['epochs'], batch_size=best['batch_size'])

test_preds = full_model.predict(x_test_std)
ids = [str(i) for i in range(1, len(test_preds) + 1)]
submission_path = Path('submission_hparam_search.csv')
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

nbf.write(nb, 'ift3395_hparam_search.ipynb')
print('Notebook created: ift3395_hparam_search.ipynb')
