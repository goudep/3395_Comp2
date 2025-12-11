import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(dedent("""
# IFT3395 Competition 2 - Early Stopping CV Ensemble (NumPy)

目标：在每个折内使用早停保存最佳模型，结合交叉验证集成以提升 Kaggle 成绩。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path('data')
SEED = 888
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

def kfold_indices(n_samples: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = [(n_samples + i) // k for i in range(k)]
    current = 0
    folds = []
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits

def simple_split(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    val_size = int(len(x) * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]

x_flat = flatten_and_normalize(images)
x_test_flat = flatten_and_normalize(test_images)
scaler = StandardScaler().fit(x_flat)
x_std = scaler.transform(x_flat)
x_test_std = scaler.transform(x_test_flat)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class EarlyStopMLP:
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 640,
        lr: float = 0.05,
        l2: float = 8e-5,
        dropout: float = 0.1,
        seed: int = 0,
    ):
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

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
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

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray, class_weights: np.ndarray):
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
        grads = {}
        grads['w2'] = h1.T @ grad_logits + self.l2 * self.w2
        grads['b2'] = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ self.w2.T
        grad_hidden[z1 <= 0.0] = 0.0
        grads['w1'] = x_cache.T @ grad_hidden + self.l2 * self.w1
        grads['b1'] = grad_hidden.sum(axis=0)
        return loss, grads

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        class_weights: np.ndarray,
        epochs: int = 400,
        batch_size: int = 128,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        patience: int = 30,
    ) -> Dict[str, List[float]]:
        history = {'train_acc': [], 'val_acc': []}
        best_state = self.get_state()
        best_metric = -np.inf
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
                yb = y[idx]
                _, grads = self._loss_and_grads(xb, yb, class_weights)
                self.w1 -= self.lr * grads['w1']
                self.b1 -= self.lr * grads['b1']
                self.w2 -= self.lr * grads['w2']
                self.b2 -= self.lr * grads['b2']
            train_acc = self.accuracy(x, y)
            history['train_acc'].append(train_acc)
            if val_data is not None:
                val_acc = self.accuracy(*val_data)
                history['val_acc'].append(val_acc)
                if val_acc > best_metric + 1e-4:
                    best_metric = val_acc
                    best_state = self.get_state()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
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

    def clone(self):
        new = EarlyStopMLP(self.w1.shape[0], self.w2.shape[1], self.hidden_dim, self.base_lr, self.l2, self.dropout)
        new.load_state(self.get_state())
        return new
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
K = 5
splits = kfold_indices(len(x_std), K, SEED)
fold_models: List[EarlyStopMLP] = []
metrics = []
params = {'hidden_dim': 640, 'lr': 0.055, 'l2': 7e-5, 'dropout': 0.12, 'epochs': 360, 'batch_size': 96, 'patience': 40}

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"Fold {fold + 1}/{K}")
    model = EarlyStopMLP(
        input_dim=x_std.shape[1],
        num_classes=num_classes,
        hidden_dim=params['hidden_dim'],
        lr=params['lr'],
        l2=params['l2'],
        dropout=params['dropout'],
        seed=SEED + fold
    )
    history = model.fit(
        x_std[train_idx],
        labels[train_idx],
        class_weights=class_weights,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        val_data=(x_std[val_idx], labels[val_idx]),
        patience=params['patience']
    )
    train_acc = history['train_acc'][-1]
    val_acc = history['val_acc'][-1]
    print(f"  train acc={train_acc:.4f}, val acc={val_acc:.4f}")
    metrics.append(val_acc)
    fold_models.append(model.clone())

print('Mean val acc:', np.mean(metrics))
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def ensemble_predict(models: List[EarlyStopMLP], x: np.ndarray) -> np.ndarray:
    probs = np.stack([m.predict_proba(x) for m in models], axis=0)
    return np.argmax(probs.mean(axis=0), axis=1)

val_preds = []
val_targets = []
for _, (_, val_idx) in enumerate(splits):
    preds = ensemble_predict(fold_models, x_std[val_idx])
    val_preds.append(preds)
    val_targets.append(labels[val_idx])
val_preds = np.concatenate(val_preds)
val_targets = np.concatenate(val_targets)
print('Ensemble stacked val acc:', (val_preds == val_targets).mean())
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## 全量训练 + 早停
使用相同超参但在全量训练集上划出 10% 作为内部验证，用于早停，然后与折模型一起对测试集进行集成预测。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
full_model = EarlyStopMLP(
    input_dim=x_std.shape[1],
    num_classes=num_classes,
    hidden_dim=params['hidden_dim'],
    lr=params['lr'],
    l2=params['l2'],
    dropout=params['dropout'],
    seed=SEED * 2
)

train_subset, val_subset, y_train_subset, y_val_subset = simple_split(x_std, labels, val_ratio=0.1, seed=SEED * 3)

full_history = full_model.fit(
    train_subset,
    y_train_subset,
    class_weights=class_weights,
    epochs=params['epochs'],
    batch_size=params['batch_size'],
    val_data=(val_subset, y_val_subset),
    patience=params['patience']
)
print(f"Full-model internal val acc: {full_history['val_acc'][-1]:.4f}")

all_models = fold_models + [full_model]
probs = np.stack([m.predict_proba(x_test_std) for m in all_models], axis=0)
ensemble_probs = probs.mean(axis=0)
test_preds = np.argmax(ensemble_probs, axis=1)

ids = [str(i) for i in range(1, len(test_preds) + 1)]
submission_path = Path('submission_earlystop_cv.csv')
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

nbf.write(nb, 'ift3395_earlystop_cv.ipynb')
print('Notebook created: ift3395_earlystop_cv.ipynb')
