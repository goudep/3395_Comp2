import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(dedent("""
# IFT3395 Competition 2 - CV Ensemble Notebook

目标：在仅使用 NumPy/标准库的前提下，通过交叉验证 + 类别重加权 + 学习率调度 + 模型集成，稳定超过 Kaggle baseline 0.455。
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## 方法概览
- 计算类别权重，缓解样本不平衡。
- 使用带 ReLU 的两层感知机，加入特征层 Dropout 与余弦学习率调度。
- 5 折交叉验证：对每折训练独立模型，验证集性能取均值以判定是否超过 baseline。
- 预测阶段对折内模型进行软投票，随后在全数据上重训并生成提交文件（`ID,Label`，ID 从 1 开始）。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path('data')
SEED = 123
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
print(f"Train images: {images.shape}, labels: {labels.shape}, classes: {num_classes}")
print(f"Test images:  {test_images.shape}")
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
unique, counts = np.unique(labels, return_counts=True)
print('Class distribution:')
for u, c in zip(unique, counts):
    print(f"  class {u}: {c} ({c / len(labels):.2%})")
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
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits

x_flat = flatten_and_normalize(images)
x_test_flat = flatten_and_normalize(test_images)
scaler = StandardScaler().fit(x_flat)
x_std = scaler.transform(x_flat)
x_test_std = scaler.transform(x_test_flat)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class NeuralNetClassifier:
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 640, lr: float = 0.05, l2: float = 5e-5, feature_dropout: float = 0.1, seed: int = 0):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.base_lr = lr
        self.l2 = l2
        self.feature_dropout = feature_dropout
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.015, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.015, size=(hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, x: np.ndarray, train: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if train and self.feature_dropout > 0.0:
            mask = (np.random.random(size=x.shape) >= self.feature_dropout).astype(np.float32)
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
        l2_pen = 0.5 * self.l2 * (np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2))
        loss += l2_pen
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

    def fit(self, x: np.ndarray, y: np.ndarray, *, class_weights: np.ndarray, epochs: int = 320, batch_size: int = 128, val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
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
                loss, grads = self._loss_and_grads(xb, yb, class_weights)
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

    def copy(self) -> 'NeuralNetClassifier':
        clone = NeuralNetClassifier(input_dim=self.w1.shape[0], num_classes=self.w2.shape[1], hidden_dim=self.hidden_dim, lr=self.base_lr, l2=self.l2, feature_dropout=self.feature_dropout)
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        return clone
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
K = 5
splits = kfold_indices(len(x_std), K, SEED)
fold_models: List[NeuralNetClassifier] = []
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"Fold {fold + 1}/{K}: train {len(train_idx)}, val {len(val_idx)}")
    model = NeuralNetClassifier(input_dim=x_std.shape[1], num_classes=num_classes, hidden_dim=640, lr=0.06, l2=8e-5, feature_dropout=0.15, seed=SEED + fold)
    history = model.fit(
        x_std[train_idx],
        labels[train_idx],
        class_weights=class_weights,
        epochs=260,
        batch_size=96,
        val_data=(x_std[val_idx], labels[val_idx])
    )
    val_acc = history['val_acc'][-1]
    train_acc = history['train_acc'][-1]
    fold_metrics.append({'train': train_acc, 'val': val_acc})
    print(f"  train acc={train_acc:.4f}, val acc={val_acc:.4f}")
    fold_models.append(model.copy())

mean_val = np.mean([m['val'] for m in fold_metrics])
print('Mean validation accuracy:', mean_val)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def ensemble_predict(models: List[NeuralNetClassifier], x: np.ndarray) -> np.ndarray:
    probs = np.stack([m.predict_proba(x) for m in models], axis=0)
    avg = probs.mean(axis=0)
    return np.argmax(avg, axis=1)

val_preds = []
val_targets = []
for fold, (train_idx, val_idx) in enumerate(splits):
    preds = ensemble_predict(fold_models, x_std[val_idx])
    val_preds.append(preds)
    val_targets.append(labels[val_idx])
val_preds = np.concatenate(val_preds)
val_targets = np.concatenate(val_targets)
ensemble_val_acc = (val_preds == val_targets).mean()
print(f"Ensemble val accuracy (stacked folds): {ensemble_val_acc:.4f}")
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## 全量训练 + 提交文件
使用最优超参在全部训练样本上重训一遍模型，随后与折内模型一起对测试集进行软投票，生成 `submission.csv`（列名 `ID,Label`，ID 从 1 开始）。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
full_model = NeuralNetClassifier(input_dim=x_std.shape[1], num_classes=num_classes, hidden_dim=640, lr=0.06, l2=8e-5, feature_dropout=0.1, seed=SEED)
full_model.fit(x_std, labels, class_weights=class_weights, epochs=320, batch_size=96)

all_models = fold_models + [full_model]
probs_stack = np.stack([m.predict_proba(x_test_std) for m in all_models], axis=0)
test_probs = probs_stack.mean(axis=0)
test_preds = np.argmax(test_probs, axis=1)

ids = [str(i) for i in range(1, len(test_preds) + 1)]
submission_path = Path('submission_cv_ensemble.csv')
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

nbf.write(nb, 'ift3395_cv_ensemble.ipynb')
print('Notebook created: ift3395_cv_ensemble.ipynb')
