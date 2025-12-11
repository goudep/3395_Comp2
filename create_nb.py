import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(dedent("""
# IFT3395 Competition 2 - NumPy Baseline
本 notebook 提供符合里程碑要求的从零实现：仅依赖 NumPy/标准库完成数据探索、逻辑回归/MLP、决策树/随机森林、核感知机，并生成可上传 Kaggle 的提交文件。
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## 数据与约束
- 训练/测试数据来自 `data/train_data.pkl` 与 `data/test_data.pkl`，分别包含彩色 28×28 图像及训练标签。
- 仅使用 NumPy 与 Python 标准库，不调用 scikit-learn 等现成 ML 库。
- 通过 Notebook 中的模块化代码，可分别训练多种模型并在 Kaggle 上验证超越 baseline。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path('data')
SEED = 42
rng = np.random.default_rng(SEED)
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
print('类别分布:')
for u, c in zip(unique, counts):
    print(f"  class {u}: {c} ({c / len(labels):.2%})")
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
        self.std_ = x.std(axis=0, keepdims=True) + 1e-6
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
x_flat_std = scaler.transform(x_flat)
x_test_std = scaler.transform(x_test_flat)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class NeuralNetClassifier:
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 0, lr: float = 0.05, l2: float = 1e-4, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.l2 = l2
        rng = np.random.default_rng(seed)
        if hidden_dim > 0:
            self.w1 = rng.normal(0.0, 0.02, size=(input_dim, hidden_dim))
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.w2 = rng.normal(0.0, 0.02, size=(hidden_dim, num_classes))
            self.b2 = np.zeros(num_classes, dtype=np.float32)
        else:
            self.w = rng.normal(0.0, 0.02, size=(input_dim, num_classes))
            self.b = np.zeros(num_classes, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        if self.hidden_dim > 0:
            z1 = x @ self.w1 + self.b1
            h1 = np.maximum(0.0, z1)
            logits = h1 @ self.w2 + self.b2
            return logits, (x, z1, h1)
        logits = x @ self.w + self.b
        return logits, (x,)

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        logits, cache = self._forward(x)
        probs = self._softmax(logits)
        y_onehot = np.eye(probs.shape[1], dtype=np.float32)[y]
        batch = x.shape[0]
        loss = -np.sum(y_onehot * np.log(probs + 1e-8)) / batch
        l2_pen = 0.0
        grads: Dict[str, np.ndarray] = {}
        if self.hidden_dim > 0:
            l2_pen += np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2)
        else:
            l2_pen += np.sum(self.w * self.w)
        loss += self.l2 * l2_pen / 2.0
        grad_logits = (probs - y_onehot) / batch
        if self.hidden_dim > 0:
            x_cache, z1, h1 = cache
            grads['w2'] = h1.T @ grad_logits + self.l2 * self.w2
            grads['b2'] = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ self.w2.T
            grad_hidden[z1 <= 0.0] = 0.0
            grads['w1'] = x_cache.T @ grad_hidden + self.l2 * self.w1
            grads['b1'] = grad_hidden.sum(axis=0)
        else:
            (x_cache,) = cache
            grads['w'] = x_cache.T @ grad_logits + self.l2 * self.w
            grads['b'] = grad_logits.sum(axis=0)
        return loss, grads

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 200, batch_size: int = 128, val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {'train_loss': [], 'train_acc': []}
        if val_data is not None:
            history['val_loss'] = []
            history['val_acc'] = []
        num_samples = x.shape[0]
        for _ in range(epochs):
            idx = np.random.permutation(num_samples)
            xb = x[idx]
            yb = y[idx]
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_x = xb[start:end]
                batch_y = yb[start:end]
                _, grads = self._loss_and_grads(batch_x, batch_y)
                if self.hidden_dim > 0:
                    self.w1 -= self.lr * grads['w1']
                    self.b1 -= self.lr * grads['b1']
                    self.w2 -= self.lr * grads['w2']
                    self.b2 -= self.lr * grads['b2']
                else:
                    self.w -= self.lr * grads['w']
                    self.b -= self.lr * grads['b']
            train_loss, _ = self._loss_and_grads(x, y)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(self.accuracy(x, y))
            if val_data is not None:
                xv, yv = val_data
                val_loss, _ = self._loss_and_grads(xv, yv)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(self.accuracy(xv, yv))
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
VAL_RATIO = 0.15
x_train, x_val, y_train, y_val = train_val_split(x_flat_std, labels, VAL_RATIO, SEED)

mlp = NeuralNetClassifier(input_dim=x_flat_std.shape[1], num_classes=num_classes, hidden_dim=512, lr=0.05, l2=1e-4, seed=SEED)
history = mlp.fit(x_train, y_train, epochs=250, batch_size=128, val_data=(x_val, y_val))
print(f"Train acc: {history['train_acc'][-1]:.4f}")
print(f"Val   acc: {history['val_acc'][-1]:.4f}")
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## 决策树 / 随机森林
下面实现一个轻量级的基于基尼不纯度的树模型，并通过装袋组成随机森林。为了控制复杂度，在每次划分时只随机抽取部分特征与阈值候选。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class SimpleDecisionTree:
    def __init__(self, max_depth: int = 6, min_samples: int = 20, feature_subsample: int = 256, thresholds: int = 10, seed: int = 0):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_subsample = feature_subsample
        self.thresholds = thresholds
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.root = None

    @staticmethod
    def gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs * probs)

    def best_split(self, x: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float, float]]:
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        base_impurity = self.gini(y)
        feature_indices = self.rng.choice(x.shape[1], size=min(self.feature_subsample, x.shape[1]), replace=False)
        for feat in feature_indices:
            col = x[:, feat]
            if np.all(col == col[0]):
                continue
            percentiles = np.linspace(0.0, 100.0, num=self.thresholds + 2, endpoint=True)[1:-1]
            th_values = np.unique(np.percentile(col, percentiles))
            for thr in th_values:
                left_mask = col <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples or right_mask.sum() < self.min_samples:
                    continue
                left_imp = self.gini(y[left_mask])
                right_imp = self.gini(y[right_mask])
                impurity = (left_mask.sum() * left_imp + right_mask.sum() * right_imp) / len(y)
                gain = base_impurity - impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = float(thr)
        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain

    def build(self, x: np.ndarray, y: np.ndarray, depth: int = 0):
        node = {'depth': depth, 'prediction': int(np.bincount(y).argmax())}
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples:
            return node
        split = self.best_split(x, y)
        if split is None:
            return node
        feat, thr, _ = split
        left_mask = x[:, feat] <= thr
        right_mask = ~left_mask
        node['feature'] = feat
        node['threshold'] = thr
        node['left'] = self.build(x[left_mask], y[left_mask], depth + 1)
        node['right'] = self.build(x[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.root = self.build(x, y, 0)
        return self

    def _predict_one(self, node, sample):
        if 'feature' not in node:
            return node['prediction']
        if sample[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], sample)
        return self._predict_one(node['right'], sample)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError('Tree not fitted')
        return np.array([self._predict_one(self.root, sample) for sample in x], dtype=int)

class RandomForest:
    def __init__(self, n_estimators: int = 15, max_depth: int = 6, feature_subsample: int = 256, min_samples: int = 20, seed: int = 0):
        self.trees: List[SimpleDecisionTree] = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample = feature_subsample
        self.min_samples = min_samples
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for i in range(self.n_estimators):
            indices = rng.integers(0, len(x), size=len(x))
            tree = SimpleDecisionTree(max_depth=self.max_depth, min_samples=self.min_samples, feature_subsample=self.feature_subsample, seed=self.seed + i)
            tree.fit(x[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = np.stack([tree.predict(x) for tree in self.trees], axis=0)
        votes = []
        for i in range(x.shape[0]):
            counts = np.bincount(preds[:, i], minlength=num_classes)
            votes.append(np.argmax(counts))
        return np.array(votes, dtype=int)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
rf = RandomForest(n_estimators=20, max_depth=7, feature_subsample=384, min_samples=10, seed=SEED)
rf.fit(x_train, y_train)
rf_train_acc = rf.accuracy(x_train, y_train)
rf_val_acc = rf.accuracy(x_val, y_val)
print(f"RF train acc: {rf_train_acc:.4f}")
print(f"RF val   acc: {rf_val_acc:.4f}")
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## 核感知机（Kernel Perceptron）
实现一个多分类核感知机（使用 RBF 核），同样只依靠 NumPy，适合集合少量非线性特征并保持实现简洁。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
class KernelPerceptron:
    def __init__(self, num_classes: int, gamma: float = 1e-3, epochs: int = 5):
        self.num_classes = num_classes
        self.gamma = gamma
        self.epochs = epochs
        self.alpha = None
        self.support_x = None

    def _rbf(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1_sq = np.sum(x1 * x1, axis=1, keepdims=True)
        x2_sq = np.sum(x2 * x2, axis=1)
        dist = x1_sq - 2 * (x1 @ x2.T) + x2_sq
        return np.exp(-self.gamma * dist)

    def fit(self, x: np.ndarray, y: np.ndarray):
        n = len(x)
        self.support_x = x
        self.alpha = np.zeros((n, self.num_classes), dtype=np.float32)
        gram = self._rbf(x, x)
        for _ in range(self.epochs):
            for i in range(n):
                scores = self.alpha.T @ gram[:, i]
                pred = int(np.argmax(scores))
                if pred != y[i]:
                    self.alpha[i, y[i]] += 1.0
                    self.alpha[i, pred] -= 1.0
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.support_x is None or self.alpha is None:
            raise ValueError('Model not fitted')
        k = self._rbf(self.support_x, x)
        scores = self.alpha.T @ k
        return np.argmax(scores, axis=0)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
subset = 600  # 样本数越多，核方法越慢，可酌情调整
indices = np.random.default_rng(SEED).choice(len(x_train), size=subset, replace=False)
kp = KernelPerceptron(num_classes=num_classes, gamma=5e-4, epochs=3)
kp.fit(x_train[indices], y_train[indices])
kp_val_acc = kp.accuracy(x_val, y_val)
print(f"Kernel Perceptron val acc: {kp_val_acc:.4f}")
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## 生成 Kaggle 提交文件
默认使用验证表现最稳定的 MLP/逻辑回归模型。若需要与官方 `sample_submission.csv` 完全一致的 ID 顺序，可将其放到 `data/` 目录，代码会自动加载对应的 `ID` 列；否则退化为 1..N 的顺序。
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
# 使用全部训练样本重新拟合最佳模型
full_model = NeuralNetClassifier(input_dim=x_flat_std.shape[1], num_classes=num_classes, hidden_dim=512, lr=0.05, l2=1e-4, seed=SEED)
full_model.fit(x_flat_std, labels, epochs=300, batch_size=128)
full_preds = full_model.predict(x_test_std)

sample_ids = None
sample_path = DATA_DIR / 'sample_submission.csv'
if sample_path.exists():
    with sample_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        sample_ids = [row[reader.fieldnames[0]] for row in reader]

if sample_ids is not None:
    ids = sample_ids
else:
    ids = [str(i) for i in range(1, len(full_preds) + 1)]

submission_path = Path('submission.csv')
with submission_path.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'label'])
    for idx, label in zip(ids, full_preds):
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
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python'
    }
}

nbf.write(nb, 'ift3395_baseline.ipynb')
print('Notebook created: ift3395_baseline.ipynb')
