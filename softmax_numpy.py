"""
IFT3395 Competition 2 baseline built only with NumPy.

This script trains a multinomial logistic regression (softmax classifier)
from scratch and generates a Kaggle-ready submission file.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StandardScaler:
    """Simple feature-wise standardization."""

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before calling transform().")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    """Reshape images to 2D array and scale to [0, 1]."""
    x = images.astype(np.float32) / 255.0
    return x.reshape(len(images), -1)


class NeuralNetClassifier:
    """Either plain softmax or a one-hidden-layer MLP depending on hidden_dim."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float,
        l2: float,
        seed: int,
        hidden_dim: int,
    ):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.l2 = l2
        self.hidden_dim = hidden_dim
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
            cache: Tuple[np.ndarray, ...] = (x, z1, h1)
        else:
            logits = x @ self.w + self.b
            cache = (x,)
        return logits, cache

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        logits, cache = self._forward(x)
        probs = self._softmax(logits)
        y_onehot = np.eye(probs.shape[1], dtype=np.float32)[y]
        batch_size = x.shape[0]
        loss = -np.sum(y_onehot * np.log(probs + 1e-8)) / batch_size
        l2_penalty = 0.0
        grads: Dict[str, np.ndarray] = {}
        if self.hidden_dim > 0:
            l2_penalty += np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2)
        else:
            l2_penalty += np.sum(self.w * self.w)
        loss += self.l2 * l2_penalty / 2.0
        grad_logits = (probs - y_onehot) / batch_size
        if self.hidden_dim > 0:
            _, z1, h1 = cache
            grads["w2"] = h1.T @ grad_logits + self.l2 * self.w2
            grads["b2"] = grad_logits.sum(axis=0)
            grad_h = grad_logits @ self.w2.T
            grad_h[z1 <= 0.0] = 0.0
            x_cache = cache[0]
            grads["w1"] = x_cache.T @ grad_h + self.l2 * self.w1
            grads["b1"] = grad_h.sum(axis=0)
        else:
            x_cache = cache[0]
            grads["w"] = x_cache.T @ grad_logits + self.l2 * self.w
            grads["b"] = grad_logits.sum(axis=0)
        return loss, grads

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"train_loss": [], "train_acc": []}
        if val_data is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        num_samples = x.shape[0]
        for _ in range(epochs):
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                xb = x_shuffled[start:end]
                yb = y_shuffled[start:end]
                loss, grads = self._loss_and_grads(xb, yb)
                if self.hidden_dim > 0:
                    self.w1 -= self.lr * grads["w1"]
                    self.b1 -= self.lr * grads["b1"]
                    self.w2 -= self.lr * grads["w2"]
                    self.b2 -= self.lr * grads["b2"]
                else:
                    self.w -= self.lr * grads["w"]
                    self.b -= self.lr * grads["b"]

            train_loss, _ = self._loss_and_grads(x, y)
            train_acc = self.accuracy(x, y)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if val_data is not None:
                xv, yv = val_data
                val_loss, _ = self._loss_and_grads(xv, yv)
                val_acc = self.accuracy(xv, yv)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(x) == y).mean()


def train_val_split(
    x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    val_size = int(len(x) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


def load_pickle(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        return pickle.load(f)


def save_submission(ids: np.ndarray, preds: np.ndarray, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label"])
        for idx, label in zip(ids, preds):
            writer.writerow([int(idx), int(label)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy softmax baseline for IFT3395 competition 2.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Folder containing the pkl files.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for gradient descent.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of training data used for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden units for the optional MLP head.")
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"), help="Path to save prediction csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_data = load_pickle(args.data_dir / "train_data.pkl")
    test_data = load_pickle(args.data_dir / "test_data.pkl")

    x = flatten_and_normalize(train_data["images"])
    y = train_data["labels"].reshape(-1).astype(int)
    x_test = flatten_and_normalize(test_data["images"])

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    x_train, x_val, y_train, y_val = train_val_split(x, y, args.val_ratio, args.seed)

    model = NeuralNetClassifier(
        input_dim=x.shape[1],
        num_classes=len(np.unique(y)),
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        hidden_dim=max(0, args.hidden_dim),
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_data=(x_val, y_val),
    )

    train_acc = history["train_acc"][-1]
    val_acc = history["val_acc"][-1]
    print(f"Final train acc: {train_acc:.4f}")
    print(f"Final val acc:   {val_acc:.4f}")

    full_model = NeuralNetClassifier(
        input_dim=x.shape[1],
        num_classes=len(np.unique(y)),
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        hidden_dim=max(0, args.hidden_dim),
    )
    full_model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size)
    preds = full_model.predict(x_test)
    save_submission(np.arange(len(preds)), preds, args.submission)
    print(f"Kaggle submission saved to {args.submission}")


if __name__ == "__main__":
    main()

