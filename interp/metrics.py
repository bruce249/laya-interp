"""Logit geometry, calibration and statistics. Inputs are raw option logits in option order."""

from __future__ import annotations

import numpy as np


def centered(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return s - s.mean(-1, keepdims=True)


def norm(s: np.ndarray) -> np.ndarray:
    return np.linalg.norm(centered(s), axis=-1)


def direction(s: np.ndarray) -> np.ndarray:
    z = centered(s)
    return z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12, None)


def rotation(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    """Angle in degrees between the centered logit directions."""
    cos = (direction(s0) * direction(s1)).sum(-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def log_scale(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    return np.log(np.clip(norm(s1), 1e-12, None) / np.clip(norm(s0), 1e-12, None))


def softmax(s: np.ndarray, t: float | np.ndarray = 1.0) -> np.ndarray:
    z = np.asarray(s, dtype=np.float64) / np.asarray(t)[..., None] if np.ndim(t) else np.asarray(s) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def margin(s: np.ndarray) -> np.ndarray:
    top2 = np.sort(np.asarray(s), axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    label = np.asarray(label).astype(int)
    if label.min() == label.max():
        return float("nan")
    return float(roc_auc_score(label, score))


def bootstrap_ci(values, stat=np.mean, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """(estimate, lo, hi) for `stat` over the first axis of `values`."""
    values = np.asarray(values)
    if len(values) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    return float(stat(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def ci_dict(values, stat=np.mean, **kw) -> dict:
    est, lo, hi = bootstrap_ci(values, stat, **kw)
    return {"est": est, "lo": lo, "hi": hi, "n": int(len(values))}


def recovery(patched, corrupt, clean):
    denom = np.asarray(clean) - np.asarray(corrupt)
    return (np.asarray(patched) - np.asarray(corrupt)) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
